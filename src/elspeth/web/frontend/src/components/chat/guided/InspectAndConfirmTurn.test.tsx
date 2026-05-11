// ============================================================================
// InspectAndConfirmTurn — wire-response contract regression coverage.
//
// Pins three wire-response contracts and one UI-state contract:
//   "Looks right" submit  → edited_values = payload.observed verbatim,
//                           chosen=null, custom_inputs=null, all-other-fields null
//   "Apply edits" submit  → edited_values.columns reflects user renames/removals;
//                           samples and warnings pass through unchanged from payload.observed
//   Editor sub-state      → "Edit columns..." opens the editor (rename inputs + remove
//                           buttons per column); Cancel returns to inspect view without
//                           submitting
//   DOM ID isolation      → two simultaneous instances with overlapping column names
//                           produce distinct element IDs for edit-mode inputs
//
// The GuidedRespondRequest shape is the unit under test; these tests will
// catch any future refactor that silently drops a null field, changes which
// fields are populated on each submit path, or corrupts samples/warnings
// pass-through on the edit path.  Tasks 7.4-7.7 will replicate this
// regression-intent structure with their own wire contracts.
// ============================================================================

import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { InspectAndConfirmTurn } from "./InspectAndConfirmTurn";
import { nullResponse } from "@/test/guided-fixtures";
import type { InspectAndConfirmPayload } from "@/types/guided";
import type { GuidedRespondRequest } from "@/types/guided";

// ── Fixtures ─────────────────────────────────────────────────────────────────

const PAYLOAD_WITH_WARNINGS: InspectAndConfirmPayload = {
  observed: {
    columns: ["name", "age", "city"],
    samples: [
      { name: "Alice", age: 30, city: "London" },
      { name: "Bob", age: 25, city: "Paris" },
    ],
    warnings: ["Column 'age' contains mixed numeric and string values."],
  },
};

const PAYLOAD_NO_WARNINGS: InspectAndConfirmPayload = {
  observed: {
    columns: ["id", "value"],
    samples: [{ id: "x1", value: 42 }],
    warnings: [],
  },
};

// Sparse sample: 'value' key missing from second row — valid per wire schema.
const PAYLOAD_SPARSE: InspectAndConfirmPayload = {
  observed: {
    columns: ["id", "value"],
    samples: [
      { id: "x1", value: "hello" },
      { id: "x2" }, // 'value' absent
    ],
    warnings: [],
  },
};

// ── Inspect view: column headers ──────────────────────────────────────────────

describe("InspectAndConfirmTurn — column headers", () => {
  it("renders every column as a <th>", () => {
    render(<InspectAndConfirmTurn payload={PAYLOAD_WITH_WARNINGS} onSubmit={vi.fn()} />);
    const headers = screen.getAllByRole("columnheader");
    const headerTexts = headers.map((h) => h.textContent);
    expect(headerTexts).toEqual(["name", "age", "city"]);
  });
});

// ── Inspect view: sample rows ─────────────────────────────────────────────────

describe("InspectAndConfirmTurn — sample rows", () => {
  it("renders one row per sample with all cell values", () => {
    render(<InspectAndConfirmTurn payload={PAYLOAD_WITH_WARNINGS} onSubmit={vi.fn()} />);
    // Each cell value is visible in the document
    expect(screen.getByText("Alice")).toBeTruthy();
    expect(screen.getByText("30")).toBeTruthy();
    expect(screen.getByText("London")).toBeTruthy();
    expect(screen.getByText("Bob")).toBeTruthy();
    expect(screen.getByText("25")).toBeTruthy();
    expect(screen.getByText("Paris")).toBeTruthy();
  });

  it("renders two <tr> elements in <tbody> for two samples", () => {
    const { container } = render(
      <InspectAndConfirmTurn payload={PAYLOAD_WITH_WARNINGS} onSubmit={vi.fn()} />,
    );
    const tbody = container.querySelector("tbody");
    expect(tbody).toBeTruthy();
    const rows = tbody!.querySelectorAll("tr");
    expect(rows).toHaveLength(2);
  });

  it("missing sample key renders as empty string (not 'undefined')", () => {
    const { container } = render(
      <InspectAndConfirmTurn payload={PAYLOAD_SPARSE} onSubmit={vi.fn()} />,
    );
    const tbody = container.querySelector("tbody");
    expect(tbody).toBeTruthy();
    const secondRow = tbody!.querySelectorAll("tr")[1];
    const cells = secondRow.querySelectorAll("td");
    expect(cells[0].textContent).toBe("x2");
    expect(cells[1].textContent).toBe(""); // missing key → empty string
  });
});

// ── Inspect view: warnings ────────────────────────────────────────────────────

describe("InspectAndConfirmTurn — warnings", () => {
  it("renders each warning text when warnings are non-empty", () => {
    render(<InspectAndConfirmTurn payload={PAYLOAD_WITH_WARNINGS} onSubmit={vi.fn()} />);
    expect(
      screen.getByText("Column 'age' contains mixed numeric and string values."),
    ).toBeTruthy();
  });

  it("renders warnings in a labelled aside (accessible to screen readers)", () => {
    const { container } = render(
      <InspectAndConfirmTurn payload={PAYLOAD_WITH_WARNINGS} onSubmit={vi.fn()} />,
    );
    const aside = container.querySelector("aside");
    expect(aside).toBeTruthy();
    expect(aside!.getAttribute("aria-label")).toBeTruthy();
  });

  it("does not render a warnings region when warnings array is empty", () => {
    const { container } = render(
      <InspectAndConfirmTurn payload={PAYLOAD_NO_WARNINGS} onSubmit={vi.fn()} />,
    );
    expect(container.querySelector("aside")).toBeNull();
  });
});

// ── Inspect view: "Looks right" submit ───────────────────────────────────────

describe("InspectAndConfirmTurn — Looks right submit", () => {
  it("clicking 'Looks right' fires onSubmit with observed shape verbatim", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<InspectAndConfirmTurn payload={PAYLOAD_WITH_WARNINGS} onSubmit={onSubmit} />);

    await user.click(screen.getByRole("button", { name: "Looks right" }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    const body: GuidedRespondRequest = onSubmit.mock.calls[0][0];
    // nullResponse() spread comes first so edited_values is not overridden by it.
    expect(body).toEqual<GuidedRespondRequest>({
      ...nullResponse(),
      chosen: null,
      custom_inputs: null,
      edited_values: {
        columns: ["name", "age", "city"],
        samples: [
          { name: "Alice", age: 30, city: "London" },
          { name: "Bob", age: 25, city: "Paris" },
        ],
        warnings: ["Column 'age' contains mixed numeric and string values."],
      },
    });
  });

  it("nullResponse() fields are explicitly null — not undefined or omitted", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<InspectAndConfirmTurn payload={PAYLOAD_NO_WARNINGS} onSubmit={onSubmit} />);

    await user.click(screen.getByRole("button", { name: "Looks right" }));

    const body: GuidedRespondRequest = onSubmit.mock.calls[0][0];
    expect(body.accepted_step_index).toBeNull();
    expect(body.edit_step_index).toBeNull();
    expect(body.control_signal).toBeNull();
  });
});

// ── Editor sub-state ──────────────────────────────────────────────────────────

describe("InspectAndConfirmTurn — editor sub-state", () => {
  it("'Edit columns...' button is visible in inspect view", () => {
    render(<InspectAndConfirmTurn payload={PAYLOAD_WITH_WARNINGS} onSubmit={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Edit columns..." })).toBeTruthy();
  });

  it("clicking 'Edit columns...' reveals an editable input per column", async () => {
    const user = userEvent.setup();
    render(<InspectAndConfirmTurn payload={PAYLOAD_WITH_WARNINGS} onSubmit={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Edit columns..." }));

    // Three columns → three text inputs
    const inputs = screen.getAllByRole("textbox");
    expect(inputs).toHaveLength(3);
    expect((inputs[0] as HTMLInputElement).value).toBe("name");
    expect((inputs[1] as HTMLInputElement).value).toBe("age");
    expect((inputs[2] as HTMLInputElement).value).toBe("city");
  });

  it("clicking 'Edit columns...' reveals a Remove button per column", async () => {
    const user = userEvent.setup();
    render(<InspectAndConfirmTurn payload={PAYLOAD_WITH_WARNINGS} onSubmit={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Edit columns..." }));

    const removeBtns = screen.getAllByRole("button", { name: "Remove" });
    expect(removeBtns).toHaveLength(3);
  });

  it("Cancel returns to inspect view without firing onSubmit", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<InspectAndConfirmTurn payload={PAYLOAD_WITH_WARNINGS} onSubmit={onSubmit} />);

    await user.click(screen.getByRole("button", { name: "Edit columns..." }));
    expect(screen.queryByRole("button", { name: "Looks right" })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    // Back to inspect view — "Looks right" is visible again
    expect(screen.getByRole("button", { name: "Looks right" })).toBeTruthy();
    expect(onSubmit).not.toHaveBeenCalled();
  });
});

// ── Edit submit: rename ───────────────────────────────────────────────────────

describe("InspectAndConfirmTurn — edit submit with rename", () => {
  it("renaming a column and applying edits sends edited columns; samples + warnings unchanged", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<InspectAndConfirmTurn payload={PAYLOAD_WITH_WARNINGS} onSubmit={onSubmit} />);

    await user.click(screen.getByRole("button", { name: "Edit columns..." }));

    // Rename "name" → "full_name"
    const inputs = screen.getAllByRole("textbox");
    await user.clear(inputs[0]);
    await user.type(inputs[0], "full_name");

    await user.click(screen.getByRole("button", { name: "Apply edits" }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    const body: GuidedRespondRequest = onSubmit.mock.calls[0][0];
    // nullResponse() spread comes first so edited_values is not overridden by it.
    expect(body).toEqual<GuidedRespondRequest>({
      ...nullResponse(),
      chosen: null,
      custom_inputs: null,
      edited_values: {
        columns: ["full_name", "age", "city"],
        // samples and warnings pass through unchanged
        samples: [
          { name: "Alice", age: 30, city: "London" },
          { name: "Bob", age: 25, city: "Paris" },
        ],
        warnings: ["Column 'age' contains mixed numeric and string values."],
      },
    });
  });
});

// ── Edit submit: remove ───────────────────────────────────────────────────────

describe("InspectAndConfirmTurn — edit submit with remove", () => {
  it("removing a column and applying edits sends columns without removed entry; samples unchanged", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<InspectAndConfirmTurn payload={PAYLOAD_WITH_WARNINGS} onSubmit={onSubmit} />);

    await user.click(screen.getByRole("button", { name: "Edit columns..." }));

    // Remove the second column ("age")
    const removeBtns = screen.getAllByRole("button", { name: "Remove" });
    await user.click(removeBtns[1]);

    await user.click(screen.getByRole("button", { name: "Apply edits" }));

    const body: GuidedRespondRequest = onSubmit.mock.calls[0][0];
    const ev = body.edited_values as { columns: string[]; samples: unknown[]; warnings: unknown[] };
    expect(ev.columns).toEqual(["name", "city"]);
    // Samples pass through unchanged from payload.observed
    expect(ev.samples).toEqual([
      { name: "Alice", age: 30, city: "London" },
      { name: "Bob", age: 25, city: "Paris" },
    ]);
  });
});

// ── DOM ID isolation (useId) ──────────────────────────────────────────────────

describe("InspectAndConfirmTurn — DOM ID isolation (useId)", () => {
  // Regression: GuidedHistory (Task 7.9) renders past turns alongside the active
  // one. Column names ("name", "age") recur across turns. Hard-coded edit-mode
  // input IDs (e.g. "col-0") would collide. useId() prefixes per-instance.
  it("two simultaneous instances in edit mode have distinct input IDs for column 0", async () => {
    const user = userEvent.setup();
    render(
      <div>
        <InspectAndConfirmTurn payload={PAYLOAD_WITH_WARNINGS} onSubmit={vi.fn()} />
        <InspectAndConfirmTurn payload={PAYLOAD_WITH_WARNINGS} onSubmit={vi.fn()} />
      </div>,
    );

    // Click "Edit columns..." on both instances
    const editBtns = screen.getAllByRole("button", { name: "Edit columns..." });
    expect(editBtns).toHaveLength(2);
    await user.click(editBtns[0]);
    await user.click(editBtns[1]);

    // Each instance renders 3 inputs; total = 6
    const inputs = screen.getAllByRole("textbox");
    expect(inputs).toHaveLength(6);

    // The first input from each instance (column 0) must have distinct IDs
    const id0 = inputs[0].id;
    const id3 = inputs[3].id;
    expect(id0).toBeTruthy();
    expect(id3).toBeTruthy();
    expect(id0).not.toBe(id3);

    // Each ID must resolve to its own input in the DOM
    expect(document.getElementById(id0)).toBe(inputs[0]);
    expect(document.getElementById(id3)).toBe(inputs[3]);
  });
});
