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
import type { SingleSelectPayload } from "@/types/guided";
import type { GuidedRespondRequest } from "@/types/guided";

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

// Helper to make the null-field assertion concise
function nullResponse(): Pick<
  GuidedRespondRequest,
  "edited_values" | "accepted_step_index" | "edit_step_index" | "control_signal"
> {
  return {
    edited_values: null,
    accepted_step_index: null,
    edit_step_index: null,
    control_signal: null,
  };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("SingleSelectTurn — option click", () => {
  it("renders the question text and one button per option", () => {
    render(<SingleSelectTurn payload={PAYLOAD_NO_CUSTOM} onSubmit={vi.fn()} />);

    expect(screen.getByText("Which data source do you want?")).toBeTruthy();
    expect(screen.getByRole("button", { name: "CSV File" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "REST API" })).toBeTruthy();
  });

  it("clicking an option fires onSubmit with chosen=[id] and all other fields null", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<SingleSelectTurn payload={PAYLOAD_NO_CUSTOM} onSubmit={onSubmit} />);

    await user.click(screen.getByRole("button", { name: "CSV File" }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    const body: GuidedRespondRequest = onSubmit.mock.calls[0][0];
    expect(body).toEqual<GuidedRespondRequest>({
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

    const body: GuidedRespondRequest = onSubmit.mock.calls[0][0];
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
    const body: GuidedRespondRequest = onSubmit.mock.calls[0][0];
    expect(body).toEqual<GuidedRespondRequest>({
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
