// ============================================================================
// MultiSelectWithCustomTurn — wire-response contract regression coverage.
//
// Pins THREE contracts:
//   1. Chip-toggle wire shape on Continue:
//        chosen=[<sorted selected ids>], custom_inputs=[<added customs>],
//        all four other fields explicitly null.
//   2. Custom-add wire shape on Continue:
//        custom_inputs reflects added strings in addition order, including
//        when chosen is empty (custom_inputs=[..], chosen=[]).
//
// Continue is disabled when both chosen and customs are empty (the widget's
// only way to surface "the user has asserted nothing" — preventing an empty
// submit that would crash the backend's required-field reads).
//
// Custom-add discipline:
//   - Empty / whitespace-only input → Add disabled.
//   - Duplicate of an existing custom OR an option ID → Add disabled.
//
// useId() pin: two simultaneous instances must have distinct element-level IDs
// AND distinct DOM nodes (the resolved-node assertion catches the
// duplicate-ID failure mode that string-distinctness alone would miss) so
// GuidedHistory replay (Task 7.9) does not produce DOM id collisions.
//
// Focus restoration on custom-chip removal (WCAG 2.4.3): pinned by the
// keyboard-transition tests at the bottom of this file. The convention
// matches InspectAndConfirmTurn (Task 7.3) — refs + useEffect + firstRunRef
// to skip initial mount. Future remove-path widgets should copy this.
//
// Escape branch coverage pins the "let source decide" path: chosen=[] and
// custom_inputs=[] leaves required fields source-decided while the backend
// preserves observed schema mode from persisted sink intent.
//
// The GuidedRespondRequest shape is the unit under test; these tests will
// catch any future refactor that silently drops a null field, swaps chosen
// and custom_inputs, or relaxes the disabled-on-empty Continue invariant.
// ============================================================================

import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MultiSelectWithCustomTurn } from "./MultiSelectWithCustomTurn";
import { nullResponse } from "@/test/guided-fixtures";
import type {
  GuidedRespondRequest,
  MultiSelectWithCustomPayload,
} from "@/types/guided";

// ── Fixtures ─────────────────────────────────────────────────────────────────

const PAYLOAD_THREE_OPTIONS_TWO_DEFAULT: MultiSelectWithCustomPayload = {
  question: "Which fields must appear in the output?",
  options: [
    { id: "name", label: "name", hint: null },
    { id: "price", label: "price", hint: "Numeric, currency-agnostic" },
    { id: "qty", label: "qty", hint: null },
  ],
  default_chosen: ["name", "price"],
  escape_label: "Let source decide (pass all fields through)",
};

const PAYLOAD_NO_DEFAULT_NO_ESCAPE: MultiSelectWithCustomPayload = {
  question: "Pick the fields you want.",
  options: [
    { id: "alpha", label: "alpha", hint: null },
    { id: "beta", label: "beta", hint: null },
  ],
  default_chosen: [],
  escape_label: null,
};

const PAYLOAD_WITH_ESCAPE_LABEL: MultiSelectWithCustomPayload = {
  question: "Pick fields.",
  options: [{ id: "x", label: "x", hint: null }],
  default_chosen: [],
  escape_label: "Let source decide",
};

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("MultiSelectWithCustomTurn — initial render", () => {
  it("renders the question text inside a fieldset legend", () => {
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_THREE_OPTIONS_TWO_DEFAULT}
        onSubmit={vi.fn()}
      />,
    );
    expect(
      screen.getByText("Which fields must appear in the output?"),
    ).toBeTruthy();
  });

  it("explains that multiple selections require Continue", () => {
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_THREE_OPTIONS_TWO_DEFAULT}
        onSubmit={vi.fn()}
      />,
    );

    expect(
      screen.getByText("Select all that apply, then press Continue."),
    ).toBeInTheDocument();
  });

  it("renders one toggle button per option", () => {
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_THREE_OPTIONS_TWO_DEFAULT}
        onSubmit={vi.fn()}
      />,
    );
    expect(screen.getByRole("button", { name: "name" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "price" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "qty" })).toBeTruthy();
  });

  it("chips for IDs in default_chosen are aria-pressed=true; others are aria-pressed=false", () => {
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_THREE_OPTIONS_TWO_DEFAULT}
        onSubmit={vi.fn()}
      />,
    );
    expect(
      screen
        .getByRole("button", { name: "name" })
        .getAttribute("aria-pressed"),
    ).toBe("true");
    expect(
      screen
        .getByRole("button", { name: "price" })
        .getAttribute("aria-pressed"),
    ).toBe("true");
    expect(
      screen.getByRole("button", { name: "qty" }).getAttribute("aria-pressed"),
    ).toBe("false");
  });

  it("renders option hint text and wires aria-describedby for non-null hints only", () => {
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_THREE_OPTIONS_TWO_DEFAULT}
        onSubmit={vi.fn()}
      />,
    );
    expect(screen.getByText("Numeric, currency-agnostic")).toBeTruthy();

    const priceBtn = screen.getByRole("button", { name: "price" });
    const describedBy = priceBtn.getAttribute("aria-describedby");
    expect(describedBy).toBeTruthy();
    const hintEl = document.getElementById(describedBy!);
    expect(hintEl).toBeTruthy();
    expect(hintEl!.textContent).toBe("Numeric, currency-agnostic");

    const nameBtn = screen.getByRole("button", { name: "name" });
    expect(nameBtn.getAttribute("aria-describedby")).toBeNull();
  });
});

describe("MultiSelectWithCustomTurn — chip toggle behaviour", () => {
  it("clicking an unpressed chip toggles it pressed", async () => {
    const user = userEvent.setup();
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_THREE_OPTIONS_TWO_DEFAULT}
        onSubmit={vi.fn()}
      />,
    );
    const qtyBtn = screen.getByRole("button", { name: "qty" });
    expect(qtyBtn.getAttribute("aria-pressed")).toBe("false");
    await user.click(qtyBtn);
    expect(qtyBtn.getAttribute("aria-pressed")).toBe("true");
  });

  it("clicking a pressed chip toggles it unpressed", async () => {
    const user = userEvent.setup();
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_THREE_OPTIONS_TWO_DEFAULT}
        onSubmit={vi.fn()}
      />,
    );
    const nameBtn = screen.getByRole("button", { name: "name" });
    expect(nameBtn.getAttribute("aria-pressed")).toBe("true");
    await user.click(nameBtn);
    expect(nameBtn.getAttribute("aria-pressed")).toBe("false");
  });
});

describe("MultiSelectWithCustomTurn — Continue submit (chip-only)", () => {
  it("Continue with default selection submits chosen=[<defaults sorted>], custom_inputs=[]", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_THREE_OPTIONS_TWO_DEFAULT}
        onSubmit={onSubmit}
      />,
    );
    await user.click(screen.getByRole("button", { name: /continue/i }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    const body: GuidedRespondRequest = onSubmit.mock.calls[0][0];
    expect(body).toEqual<GuidedRespondRequest>({
      ...nullResponse(),
      chosen: ["name", "price"],
      custom_inputs: [],
    });
  });

  it("toggle adds and removes are reflected in chosen on Continue (stable sort over option order)", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_THREE_OPTIONS_TWO_DEFAULT}
        onSubmit={onSubmit}
      />,
    );

    // Untoggle "name" (was a default), toggle "qty" on. Expected final
    // chosen = ["price", "qty"] in option order.
    await user.click(screen.getByRole("button", { name: "name" }));
    await user.click(screen.getByRole("button", { name: "qty" }));
    await user.click(screen.getByRole("button", { name: /continue/i }));

    const body: GuidedRespondRequest = onSubmit.mock.calls[0][0];
    expect(body).toEqual<GuidedRespondRequest>({
      ...nullResponse(),
      chosen: ["price", "qty"],
      custom_inputs: [],
    });
  });
});

describe("MultiSelectWithCustomTurn — custom-add", () => {
  it("Add button is disabled while the custom input is empty", () => {
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_NO_DEFAULT_NO_ESCAPE}
        onSubmit={vi.fn()}
      />,
    );
    const addBtn = screen.getByRole("button", { name: /^add$/i });
    expect(addBtn).toHaveProperty("disabled", true);
  });

  it("Add button is disabled when input contains only whitespace", async () => {
    const user = userEvent.setup();
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_NO_DEFAULT_NO_ESCAPE}
        onSubmit={vi.fn()}
      />,
    );
    const input = screen.getByRole("textbox", { name: /custom field/i });
    await user.type(input, "   ");
    const addBtn = screen.getByRole("button", { name: /^add$/i });
    expect(addBtn).toHaveProperty("disabled", true);
  });

  it("Add button is disabled when input duplicates an existing option ID", async () => {
    const user = userEvent.setup();
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_NO_DEFAULT_NO_ESCAPE}
        onSubmit={vi.fn()}
      />,
    );
    const input = screen.getByRole("textbox", { name: /custom field/i });
    await user.type(input, "alpha");
    const addBtn = screen.getByRole("button", { name: /^add$/i });
    expect(addBtn).toHaveProperty("disabled", true);
  });

  it("Add button is disabled when input duplicates an already-added custom", async () => {
    const user = userEvent.setup();
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_NO_DEFAULT_NO_ESCAPE}
        onSubmit={vi.fn()}
      />,
    );
    const input = screen.getByRole("textbox", { name: /custom field/i });
    await user.type(input, "extra");
    await user.click(screen.getByRole("button", { name: /^add$/i }));
    // After add: input cleared, custom chip rendered.
    expect(screen.getByText("extra")).toBeTruthy();
    // Re-typing the same value must keep Add disabled.
    await user.type(input, "extra");
    const addBtn = screen.getByRole("button", { name: /^add$/i });
    expect(addBtn).toHaveProperty("disabled", true);
  });

  it("typing then pressing Enter appends the trimmed value to customs", async () => {
    const user = userEvent.setup();
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_NO_DEFAULT_NO_ESCAPE}
        onSubmit={vi.fn()}
      />,
    );
    const input = screen.getByRole("textbox", { name: /custom field/i });
    await user.type(input, "  gamma  {Enter}");
    expect(screen.getByText("gamma")).toBeTruthy();
  });

  it("clicking the X on a custom chip removes it from customs", async () => {
    const user = userEvent.setup();
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_NO_DEFAULT_NO_ESCAPE}
        onSubmit={vi.fn()}
      />,
    );
    const input = screen.getByRole("textbox", { name: /custom field/i });
    await user.type(input, "delta");
    await user.click(screen.getByRole("button", { name: /^add$/i }));
    expect(screen.getByText("delta")).toBeTruthy();

    await user.click(screen.getByRole("button", { name: /remove delta/i }));
    expect(screen.queryByText("delta")).toBeNull();
  });
});

describe("MultiSelectWithCustomTurn — Continue submit (with customs)", () => {
  it("Continue submits custom_inputs in addition order alongside chosen", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_THREE_OPTIONS_TWO_DEFAULT}
        onSubmit={onSubmit}
      />,
    );

    const input = screen.getByRole("textbox", { name: /custom field/i });
    await user.type(input, "first_extra");
    await user.click(screen.getByRole("button", { name: /^add$/i }));
    await user.type(input, "second_extra");
    await user.click(screen.getByRole("button", { name: /^add$/i }));

    await user.click(screen.getByRole("button", { name: /continue/i }));

    const body: GuidedRespondRequest = onSubmit.mock.calls[0][0];
    expect(body).toEqual<GuidedRespondRequest>({
      ...nullResponse(),
      chosen: ["name", "price"],
      custom_inputs: ["first_extra", "second_extra"],
    });
  });

  it("Continue submits with chosen=[] when only customs are provided", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_NO_DEFAULT_NO_ESCAPE}
        onSubmit={onSubmit}
      />,
    );

    const input = screen.getByRole("textbox", { name: /custom field/i });
    await user.type(input, "only_custom");
    await user.click(screen.getByRole("button", { name: /^add$/i }));
    await user.click(screen.getByRole("button", { name: /continue/i }));

    const body: GuidedRespondRequest = onSubmit.mock.calls[0][0];
    expect(body).toEqual<GuidedRespondRequest>({
      ...nullResponse(),
      chosen: [],
      custom_inputs: ["only_custom"],
    });
  });
});

describe("MultiSelectWithCustomTurn — Continue disabled on empty assertion", () => {
  it("Continue is disabled when both chosen and customs are empty", () => {
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_NO_DEFAULT_NO_ESCAPE}
        onSubmit={vi.fn()}
      />,
    );
    const continueBtn = screen.getByRole("button", { name: /continue/i });
    expect(continueBtn).toHaveProperty("disabled", true);
  });

  it("clicking the disabled Continue button does NOT fire onSubmit", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_NO_DEFAULT_NO_ESCAPE}
        onSubmit={onSubmit}
      />,
    );
    await user.click(screen.getByRole("button", { name: /continue/i }));
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("Continue becomes enabled once a chip is toggled on", async () => {
    const user = userEvent.setup();
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_NO_DEFAULT_NO_ESCAPE}
        onSubmit={vi.fn()}
      />,
    );
    const continueBtn = screen.getByRole("button", { name: /continue/i });
    expect(continueBtn).toHaveProperty("disabled", true);
    await user.click(screen.getByRole("button", { name: "alpha" }));
    expect(continueBtn).toHaveProperty("disabled", false);
  });

  it("Continue becomes enabled once a custom is added", async () => {
    const user = userEvent.setup();
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_NO_DEFAULT_NO_ESCAPE}
        onSubmit={vi.fn()}
      />,
    );
    const continueBtn = screen.getByRole("button", { name: /continue/i });
    expect(continueBtn).toHaveProperty("disabled", true);
    const input = screen.getByRole("textbox", { name: /custom field/i });
    await user.type(input, "extra");
    await user.click(screen.getByRole("button", { name: /^add$/i }));
    expect(continueBtn).toHaveProperty("disabled", false);
  });
});

describe("MultiSelectWithCustomTurn — escape_label", () => {
  it("renders an escape button when escape_label is non-null", () => {
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_WITH_ESCAPE_LABEL}
        onSubmit={vi.fn()}
      />,
    );
    expect(
      screen.getByRole("button", { name: /let source decide/i }),
    ).toBeInTheDocument();
  });

  it("escape button submits empty required fields while preserving all null wire fields", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_WITH_ESCAPE_LABEL}
        onSubmit={onSubmit}
      />,
    );
    await user.click(screen.getByRole("button", { name: /let source decide/i }));

    expect(onSubmit).toHaveBeenCalledWith<GuidedRespondRequest[]>({
      ...nullResponse(),
      chosen: [],
      custom_inputs: [],
    });
  });

  it("does NOT render an escape button when escape_label is null", () => {
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_NO_DEFAULT_NO_ESCAPE}
        onSubmit={vi.fn()}
      />,
    );
    expect(screen.queryByText(/let source decide/i)).toBeNull();
  });
});

describe("MultiSelectWithCustomTurn — DOM ID isolation (useId)", () => {
  // Regression: GuidedHistory (Task 7.9) renders past turns alongside the
  // active one; option IDs and custom-input labels recur across turns.
  // Hard-coded element IDs would collide. useId() prefixes per-instance.
  //
  // Each test below pins NODE distinctness (not just string distinctness).
  // A buggy implementation that emitted two elements with the SAME id
  // would still pass `id0 !== id1` if the IDs were assigned via different
  // code paths, OR would silently produce duplicate IDs that
  // getElementById resolves to the same node (both `.toBeTruthy()` pass
  // because both queries return the same truthy node). The not.toBe()
  // comparison on the resolved nodes catches both failure modes.
  it("two simultaneous instances with the same option ids have distinct hint IDs and nodes", () => {
    render(
      <div>
        <MultiSelectWithCustomTurn
          payload={PAYLOAD_THREE_OPTIONS_TWO_DEFAULT}
          onSubmit={vi.fn()}
        />
        <MultiSelectWithCustomTurn
          payload={PAYLOAD_THREE_OPTIONS_TWO_DEFAULT}
          onSubmit={vi.fn()}
        />
      </div>,
    );
    const priceBtns = screen.getAllByRole("button", { name: "price" });
    expect(priceBtns).toHaveLength(2);
    const id0 = priceBtns[0].getAttribute("aria-describedby");
    const id1 = priceBtns[1].getAttribute("aria-describedby");
    expect(id0).toBeTruthy();
    expect(id1).toBeTruthy();
    expect(id0).not.toBe(id1);
    const node0 = document.getElementById(id0!);
    const node1 = document.getElementById(id1!);
    expect(node0).toBeTruthy();
    expect(node1).toBeTruthy();
    expect(node0).not.toBe(node1);
  });

  it("two simultaneous instances have distinct custom-input IDs and nodes", () => {
    render(
      <div>
        <MultiSelectWithCustomTurn
          payload={PAYLOAD_NO_DEFAULT_NO_ESCAPE}
          onSubmit={vi.fn()}
        />
        <MultiSelectWithCustomTurn
          payload={PAYLOAD_NO_DEFAULT_NO_ESCAPE}
          onSubmit={vi.fn()}
        />
      </div>,
    );
    const inputs = screen.getAllByRole("textbox", { name: /custom field/i });
    expect(inputs).toHaveLength(2);
    expect(inputs[0].id).not.toBe(inputs[1].id);
    expect(inputs[0].id).toBeTruthy();
    expect(inputs[1].id).toBeTruthy();
    expect(document.getElementById(inputs[0].id)).not.toBe(
      document.getElementById(inputs[1].id),
    );
  });
});

describe("MultiSelectWithCustomTurn — focus restoration on custom-chip removal (WCAG 2.4.3)", () => {
  // Removing a custom chip via the X button unmounts the focused element.
  // Without explicit focus restoration, focus drops to <body> and the
  // keyboard user loses their place. These tests pin the convention so
  // future remove-path widgets (or refactors here) maintain it.
  it("removing a non-last custom moves focus to the X button of the chip in the same slot", async () => {
    const user = userEvent.setup();
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_NO_DEFAULT_NO_ESCAPE}
        onSubmit={vi.fn()}
      />,
    );
    const input = screen.getByRole("textbox", { name: /custom field/i });
    const addBtn = screen.getByRole("button", { name: /^add$/i });

    await user.type(input, "alpha_custom");
    await user.click(addBtn);
    await user.type(input, "beta_custom");
    await user.click(addBtn);

    // Remove "alpha_custom" (the first chip). The chip that was at index 1
    // ("beta_custom") shifts into index 0 — its X button should receive focus.
    await user.click(
      screen.getByRole("button", { name: /remove alpha_custom/i }),
    );

    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: /remove beta_custom/i }),
    );
  });

  it("removing the last remaining custom moves focus to the custom-input field", async () => {
    // The input is the empty-list focus target rather than the Add button
    // because Add is disabled while the input is empty (focusing a disabled
    // button is a no-op). The input is always focusable and is the entry
    // point for adding more — sensible target either way.
    const user = userEvent.setup();
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_NO_DEFAULT_NO_ESCAPE}
        onSubmit={vi.fn()}
      />,
    );
    const input = screen.getByRole("textbox", { name: /custom field/i });
    const addBtn = screen.getByRole("button", { name: /^add$/i });

    await user.type(input, "lonely_custom");
    await user.click(addBtn);

    await user.click(
      screen.getByRole("button", { name: /remove lonely_custom/i }),
    );

    expect(document.activeElement).toBe(input);
  });

  it("removing the trailing chip in a list of two moves focus to the new last chip's X button", async () => {
    const user = userEvent.setup();
    render(
      <MultiSelectWithCustomTurn
        payload={PAYLOAD_NO_DEFAULT_NO_ESCAPE}
        onSubmit={vi.fn()}
      />,
    );
    const input = screen.getByRole("textbox", { name: /custom field/i });
    const addBtn = screen.getByRole("button", { name: /^add$/i });

    await user.type(input, "first_added");
    await user.click(addBtn);
    await user.type(input, "second_added");
    await user.click(addBtn);

    // Remove the chip at index 1 (the trailing one). Focus should fall back
    // to the X button of the new last chip (now "first_added" at index 0).
    await user.click(
      screen.getByRole("button", { name: /remove second_added/i }),
    );

    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: /remove first_added/i }),
    );
  });
});
