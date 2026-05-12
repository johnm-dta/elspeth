// src/components/chat/guided/RecipeOfferTurn.test.tsx
//
// Regression suite for RecipeOfferTurn (Task 7.7).
//
// Pinned contracts:
//   1. Recipe-name heading + slot summary + alternatives list
//      Recipe name renders as <h3>; slots render as <dl>/<dt>/<dd>; alternatives
//      render as a list when non-empty and are absent from the DOM when empty.
//   2. Two-button wire-shape contracts (Apply / Build manually)
//      "Apply recipe" sends chosen: ["accept"] + edited_values: { recipe_name, slots }.
//      "Build manually" sends chosen: ["build_manually"] + edited_values: null.
//      Both paths set all 6 GuidedRespondRequest fields explicitly.
//   3. Alternatives show/hide based on emptiness
//      Non-empty alternatives list renders; empty alternatives list is absent.
//   4. DOM-ID distinctness pin
//      Two simultaneous RecipeOfferTurn instances produce element-level IDs that
//      resolve to different DOM nodes (per-instance useId() scoping).
//
// Wire-shape notes (routes.py:1943-2023):
//   chosen: ["accept"]         -> backend reads edited_values.recipe_name + slots
//   chosen: ["build_manually"] -> backend only reads chosen; advances to STEP_3
//   Any other chosen value     -> HTTP 400

import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RecipeOfferTurn } from "./RecipeOfferTurn";
import { nullResponse } from "@/test/guided-fixtures";
import type { RecipeOfferPayload } from "@/types/guided";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const BASIC_PAYLOAD: RecipeOfferPayload = {
  recipe_name: "csv_to_jsonl",
  slots: { delimiter: ",", encoding: "utf-8" },
  alternatives: ["build_manually"],
};

const NO_ALTERNATIVES_PAYLOAD: RecipeOfferPayload = {
  recipe_name: "parquet_to_csv",
  slots: { compression: "snappy" },
  alternatives: [],
};

const COMPLEX_SLOTS_PAYLOAD: RecipeOfferPayload = {
  recipe_name: "enrich_geo",
  slots: {
    api_key: "***",
    timeout: 5,
    nested_obj: { a: 1, b: 2 },
    arr_val: ["x", "y"],
  },
  alternatives: ["build_manually"],
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function renderWidget(
  payload: RecipeOfferPayload,
  onSubmit = vi.fn(),
) {
  return render(<RecipeOfferTurn payload={payload} onSubmit={onSubmit} />);
}

// ---------------------------------------------------------------------------
// 1. Recipe-name heading + slot summary + alternatives list
// ---------------------------------------------------------------------------

describe("RecipeOfferTurn — content rendering", () => {
  it("renders recipe_name as a level-3 heading for screen-reader navigation", () => {
    renderWidget(BASIC_PAYLOAD);
    // Pins <h3> semantics -- a future "simplify back to <span>" regresses here.
    const heading = screen.getByRole("heading", { level: 3 });
    expect(heading).toHaveTextContent("csv_to_jsonl");
  });

  it("renders each slot key as a <dt>", () => {
    renderWidget(BASIC_PAYLOAD);
    expect(screen.getByText("delimiter")).toBeInTheDocument();
    expect(screen.getByText("encoding")).toBeInTheDocument();
  });

  it("renders each slot value as a <dd>", () => {
    renderWidget(BASIC_PAYLOAD);
    // Scalar string value appears directly
    expect(screen.getByText(",")).toBeInTheDocument();
    expect(screen.getByText("utf-8")).toBeInTheDocument();
  });

  it("JSON-stringifies non-scalar slot values (objects, arrays)", () => {
    renderWidget(COMPLEX_SLOTS_PAYLOAD);
    // Objects and arrays are JSON-stringified for human readability
    expect(screen.getByText('{"a":1,"b":2}')).toBeInTheDocument();
    expect(screen.getByText('["x","y"]')).toBeInTheDocument();
  });

  it("renders numeric slot values as strings", () => {
    renderWidget(COMPLEX_SLOTS_PAYLOAD);
    expect(screen.getByText("5")).toBeInTheDocument();
  });

  it("renders alternatives list when alternatives is non-empty", () => {
    renderWidget(BASIC_PAYLOAD); // alternatives: ["build_manually"]
    expect(screen.getByText("build_manually")).toBeInTheDocument();
  });

  it("does NOT render alternatives section when alternatives is empty (negative-space pin)", () => {
    renderWidget(NO_ALTERNATIVES_PAYLOAD); // alternatives: []
    // The alternatives section heading must be absent.
    expect(screen.queryByText(/alternatives/i)).not.toBeInTheDocument();
  });

  it("renders single slot correctly", () => {
    renderWidget(NO_ALTERNATIVES_PAYLOAD);
    const heading = screen.getByRole("heading", { level: 3 });
    expect(heading).toHaveTextContent("parquet_to_csv");
    expect(screen.getByText("compression")).toBeInTheDocument();
    expect(screen.getByText("snappy")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// 2. Two-button wire-shape contracts
// ---------------------------------------------------------------------------

describe("RecipeOfferTurn — Apply recipe submit", () => {
  it("clicking Apply recipe fires onSubmit with chosen: ['accept'] and echoes payload fields", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    renderWidget(BASIC_PAYLOAD, onSubmit);

    await user.click(screen.getByRole("button", { name: /apply recipe/i }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    // edited_values MUST carry recipe_name and slots so the backend can reconstruct
    // the RecipeMatch (routes.py:1965-1967). edited_values: null would cause a
    // server-side failure ("", {}) fallback path.
    expect(onSubmit).toHaveBeenCalledWith({
      ...nullResponse(), // spread FIRST; edited_values overridden below
      chosen: ["accept"],
      custom_inputs: null,
      edited_values: {
        recipe_name: "csv_to_jsonl",
        slots: { delimiter: ",", encoding: "utf-8" },
      },
    });
  });

  it("Apply recipe body contains all 6 GuidedRespondRequest fields explicitly", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    renderWidget(BASIC_PAYLOAD, onSubmit);

    await user.click(screen.getByRole("button", { name: /apply recipe/i }));

    const body = onSubmit.mock.calls[0][0];
    expect(body).toEqual({
      chosen: ["accept"],
      edited_values: {
        recipe_name: "csv_to_jsonl",
        slots: { delimiter: ",", encoding: "utf-8" },
      },
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });
    // Assert all 6 explicit keys are present (toEqual checks no extra keys too).
    const keys = Object.keys(body);
    expect(keys).toContain("chosen");
    expect(keys).toContain("edited_values");
    expect(keys).toContain("custom_inputs");
    expect(keys).toContain("accepted_step_index");
    expect(keys).toContain("edit_step_index");
    expect(keys).toContain("control_signal");
  });

  it("Apply recipe echoes complex slots (including nested objects) into edited_values", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    renderWidget(COMPLEX_SLOTS_PAYLOAD, onSubmit);

    await user.click(screen.getByRole("button", { name: /apply recipe/i }));

    const body = onSubmit.mock.calls[0][0];
    expect(body.edited_values).toEqual({
      recipe_name: "enrich_geo",
      slots: {
        api_key: "***",
        timeout: 5,
        nested_obj: { a: 1, b: 2 },
        arr_val: ["x", "y"],
      },
    });
  });
});

describe("RecipeOfferTurn — Build manually submit", () => {
  it("clicking Build manually fires onSubmit with chosen: ['build_manually'] and null edited_values", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    renderWidget(BASIC_PAYLOAD, onSubmit);

    await user.click(screen.getByRole("button", { name: /build manually/i }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit).toHaveBeenCalledWith({
      ...nullResponse(), // spread FIRST
      chosen: ["build_manually"],
      custom_inputs: null,
    });
  });

  it("Build manually body contains all 6 GuidedRespondRequest fields explicitly", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    renderWidget(BASIC_PAYLOAD, onSubmit);

    await user.click(screen.getByRole("button", { name: /build manually/i }));

    const body = onSubmit.mock.calls[0][0];
    expect(body).toEqual({
      chosen: ["build_manually"],
      edited_values: null,
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });
  });
});

describe("RecipeOfferTurn — no state pollution between submits", () => {
  it("Build manually after Apply (or vice versa) emits the correct body each time", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    renderWidget(BASIC_PAYLOAD, onSubmit);

    // First click: Apply recipe
    await user.click(screen.getByRole("button", { name: /apply recipe/i }));
    // Second click: Build manually
    await user.click(screen.getByRole("button", { name: /build manually/i }));

    expect(onSubmit).toHaveBeenCalledTimes(2);
    // First call: Apply
    expect(onSubmit.mock.calls[0][0]).toEqual({
      ...nullResponse(),
      chosen: ["accept"],
      custom_inputs: null,
      edited_values: {
        recipe_name: "csv_to_jsonl",
        slots: { delimiter: ",", encoding: "utf-8" },
      },
    });
    // Second call: Build manually -- no leftover state from Apply
    expect(onSubmit.mock.calls[1][0]).toEqual({
      ...nullResponse(),
      chosen: ["build_manually"],
      custom_inputs: null,
    });
  });

  it("Apply twice emits identical bodies (zero widget state)", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    renderWidget(BASIC_PAYLOAD, onSubmit);

    await user.click(screen.getByRole("button", { name: /apply recipe/i }));
    await user.click(screen.getByRole("button", { name: /apply recipe/i }));

    expect(onSubmit).toHaveBeenCalledTimes(2);
    expect(onSubmit.mock.calls[1][0]).toEqual(onSubmit.mock.calls[0][0]);
  });
});

// ---------------------------------------------------------------------------
// 4. DOM-ID distinctness pin (Task 7.4 I4 convention)
// ---------------------------------------------------------------------------

describe("RecipeOfferTurn — DOM-ID distinctness pin", () => {
  it("two simultaneous instances produce distinct element nodes", () => {
    const { container: c1 } = render(
      <RecipeOfferTurn payload={BASIC_PAYLOAD} onSubmit={vi.fn()} />,
    );
    const { container: c2 } = render(
      <RecipeOfferTurn payload={BASIC_PAYLOAD} onSubmit={vi.fn()} />,
    );

    // Each instance scopes element IDs via useId().
    const id0 = c1.querySelector("[id]")?.id;
    const id1 = c2.querySelector("[id]")?.id;

    expect(id0).toBeDefined();
    expect(id1).toBeDefined();
    // Different instances produce different ID strings.
    expect(id0).not.toBe(id1);

    // Resolve both IDs in the document -- must be different nodes.
    const node0 = document.getElementById(id0!);
    const node1 = document.getElementById(id1!);
    expect(node0).not.toBeNull();
    expect(node1).not.toBeNull();
    expect(node0).not.toBe(node1);
  });
});

// ---------------------------------------------------------------------------
// 5. Focus / auto-focus (negative-space pin)
// ---------------------------------------------------------------------------

describe("RecipeOfferTurn — focus management", () => {
  it("does NOT auto-focus any element on initial render", () => {
    // RecipeOfferTurn has no view transitions or collapsible regions; no
    // programmatic focus on mount (per Task 7.2 template convention).
    renderWidget(BASIC_PAYLOAD);
    expect(document.activeElement).toBe(document.body);
  });
});
