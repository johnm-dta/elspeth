// ============================================================================
// SingleSelectTurn — wire-response contract regression coverage.
//
// Pins SingleSelectTurn's wire-response contract:
//   option click   → chosen=[id], custom_inputs=null, all-other-fields null
//   custom-input submit → custom_inputs=[text], chosen=null, all-other-fields null
//   never both populated, never neither
//   hint text → rendered + aria-describedby wired; no hint → no aria-describedby
//   allow_custom=false → no custom-input control rendered
//
// The GuidedRespondRequest shape is the unit under test; these tests will
// catch any future refactor that silently drops a null field or swaps chosen
// and custom_inputs.  Tasks 7.3-7.7 will replicate this regression-intent
// structure with their own wire contracts.
// ============================================================================

import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SingleSelectTurn } from "./SingleSelectTurn";
import { nullResponse } from "@/test/guided-fixtures";
import type { SingleSelectPayload } from "@/types/guided";
import type { GuidedRespondAction } from "@/types/guided";

// ── Fixtures ─────────────────────────────────────────────────────────────────

const PAYLOAD_NO_CUSTOM: SingleSelectPayload = {
  question: "Which data source do you want?",
  options: [
    { id: "csv", label: "CSV File", hint: null },
    { id: "api", label: "REST API", hint: "Fetches data from an HTTP endpoint" },
  ],
  allow_custom: false,
};

const PAYLOAD_WITH_CUSTOM: SingleSelectPayload = {
  question: "Which transform should we use?",
  options: [{ id: "llm_classify", label: "LLM Classifier", hint: null }],
  allow_custom: true,
};

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("SingleSelectTurn — option click", () => {
  it("renders the question text and one button per option", () => {
    render(<SingleSelectTurn payload={PAYLOAD_NO_CUSTOM} onSubmit={vi.fn()} />);

    expect(screen.getByText("Which data source do you want?")).toBeTruthy();
    expect(screen.getByRole("button", { name: "CSV File" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "REST API" })).toBeTruthy();
  });

  it("explains that choosing one option advances immediately", () => {
    render(<SingleSelectTurn payload={PAYLOAD_NO_CUSTOM} onSubmit={vi.fn()} />);

    expect(
      screen.getByText("Select one. Choosing an option continues to the next step."),
    ).toBeInTheDocument();
  });

  it("clicking an option fires onSubmit with chosen=[id] and all other fields null", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<SingleSelectTurn payload={PAYLOAD_NO_CUSTOM} onSubmit={onSubmit} />);

    await user.click(screen.getByRole("button", { name: "CSV File" }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    const body: GuidedRespondAction = onSubmit.mock.calls[0][0];
    expect(body).toEqual<GuidedRespondAction>({
      chosen: ["csv"],
      custom_inputs: null,
      ...nullResponse(),
    });
  });

  it("option click with allow_custom=true still produces chosen=[id], custom_inputs=null (mutual exclusion)", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<SingleSelectTurn payload={PAYLOAD_WITH_CUSTOM} onSubmit={onSubmit} />);

    await user.click(screen.getByRole("button", { name: "LLM Classifier" }));

    const body: GuidedRespondAction = onSubmit.mock.calls[0][0];
    expect(body.chosen).toEqual(["llm_classify"]);
    expect(body.custom_inputs).toBeNull();
  });
});

describe("SingleSelectTurn — allow_custom=false", () => {
  it("does not render a custom-input control when allow_custom is false", () => {
    render(<SingleSelectTurn payload={PAYLOAD_NO_CUSTOM} onSubmit={vi.fn()} />);
    // The custom input row should be absent
    expect(screen.queryByPlaceholderText(/custom/i)).toBeNull();
    expect(screen.queryByLabelText(/custom/i)).toBeNull();
  });
});

describe("SingleSelectTurn — allow_custom=true", () => {
  it("renders a custom-input control when allow_custom is true", () => {
    render(<SingleSelectTurn payload={PAYLOAD_WITH_CUSTOM} onSubmit={vi.fn()} />);
    expect(screen.getByRole("textbox", { name: /custom/i })).toBeTruthy();
  });

  it("submit button is disabled while custom input is empty", () => {
    render(<SingleSelectTurn payload={PAYLOAD_WITH_CUSTOM} onSubmit={vi.fn()} />);
    const submitBtn = screen.getByRole("button", { name: /submit custom/i });
    expect(submitBtn).toHaveProperty("disabled", true);
  });

  it("submitting custom text fires onSubmit with custom_inputs=[text] and chosen=null", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<SingleSelectTurn payload={PAYLOAD_WITH_CUSTOM} onSubmit={onSubmit} />);

    const input = screen.getByRole("textbox", { name: /custom/i });
    await user.type(input, "my custom transform");
    await user.click(screen.getByRole("button", { name: /submit custom/i }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    const body: GuidedRespondAction = onSubmit.mock.calls[0][0];
    expect(body).toEqual<GuidedRespondAction>({
      chosen: null,
      custom_inputs: ["my custom transform"],
      ...nullResponse(),
    });
  });
});

describe("SingleSelectTurn — hint rendering", () => {
  it("renders hint text for options that have a non-null hint", () => {
    render(<SingleSelectTurn payload={PAYLOAD_NO_CUSTOM} onSubmit={vi.fn()} />);
    expect(screen.getByText("Fetches data from an HTTP endpoint")).toBeTruthy();
  });

  it("option with non-null hint has aria-describedby pointing to hint element", () => {
    render(<SingleSelectTurn payload={PAYLOAD_NO_CUSTOM} onSubmit={vi.fn()} />);
    const apiBtn = screen.getByRole("button", { name: "REST API" });
    const describedBy = apiBtn.getAttribute("aria-describedby");
    expect(describedBy).toBeTruthy();
    // The ID must resolve to the hint element in the DOM
    const hintEl = document.getElementById(describedBy!);
    expect(hintEl).toBeTruthy();
    expect(hintEl!.textContent).toBe("Fetches data from an HTTP endpoint");
  });

  it("option with null hint has no aria-describedby", () => {
    render(<SingleSelectTurn payload={PAYLOAD_NO_CUSTOM} onSubmit={vi.fn()} />);
    const csvBtn = screen.getByRole("button", { name: "CSV File" });
    expect(csvBtn.getAttribute("aria-describedby")).toBeNull();
  });
});

describe("SingleSelectTurn — DOM ID isolation (useId)", () => {
  // Regression: GuidedHistory (Task 7.9) renders past turns alongside the
  // active one; option IDs ("csv", "api") recur across turns. Hard-coded
  // hint IDs (e.g. "hint-api") would collide. useId() prefixes per-instance.
  it("two simultaneous instances with the same option ids have distinct hint IDs", () => {
    render(
      <div>
        <SingleSelectTurn payload={PAYLOAD_NO_CUSTOM} onSubmit={vi.fn()} />
        <SingleSelectTurn payload={PAYLOAD_NO_CUSTOM} onSubmit={vi.fn()} />
      </div>,
    );

    const restBtns = screen.getAllByRole("button", { name: "REST API" });
    expect(restBtns).toHaveLength(2);
    const id0 = restBtns[0].getAttribute("aria-describedby");
    const id1 = restBtns[1].getAttribute("aria-describedby");
    expect(id0).toBeTruthy();
    expect(id1).toBeTruthy();
    expect(id0).not.toBe(id1);

    // Both IDs must resolve to a unique hint element in the DOM.
    expect(document.getElementById(id0!)).toBeTruthy();
    expect(document.getElementById(id1!)).toBeTruthy();
  });

  it("two simultaneous instances with allow_custom have distinct input IDs", () => {
    render(
      <div>
        <SingleSelectTurn payload={PAYLOAD_WITH_CUSTOM} onSubmit={vi.fn()} />
        <SingleSelectTurn payload={PAYLOAD_WITH_CUSTOM} onSubmit={vi.fn()} />
      </div>,
    );

    const inputs = screen.getAllByRole("textbox", { name: /custom/i });
    expect(inputs).toHaveLength(2);
    expect(inputs[0].id).not.toBe(inputs[1].id);
    // The <label htmlFor> must point to its own input, not the sibling's
    // — this is what makes the per-label getByRole queries above work
    // correctly across multiple instances.
    expect(inputs[0].id).toBeTruthy();
    expect(inputs[1].id).toBeTruthy();
  });
});

// ── Tutorial passive mode ────────────────────────────────────────────────────

describe("SingleSelectTurn — tutorial passive mode", () => {
  it("suppresses the 'Choosing an option continues' subtext but keeps the chips", () => {
    render(
      <SingleSelectTurn payload={PAYLOAD_NO_CUSTOM} onSubmit={vi.fn()} isTutorial />,
    );
    expect(screen.queryByText(/choosing an option continues/i)).toBeNull();
    // The option chips themselves remain interactive (Send is the guided path,
    // but a learner who clicks an option still advances).
    expect(screen.getByRole("button", { name: "CSV File" })).toBeInTheDocument();
  });

  it("shows the subtext in normal (non-tutorial) mode", () => {
    render(<SingleSelectTurn payload={PAYLOAD_NO_CUSTOM} onSubmit={vi.fn()} />);
    expect(
      screen.getByText(/choosing an option continues/i),
    ).toBeInTheDocument();
  });
});
