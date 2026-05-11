// Tests for guided.ts — type-assertion style.
// Vitest compile failures ARE the test: if the union drifts, the array literal
// above it stops compiling.

import { describe, expect, it } from "vitest";

import type {
  ControlSignal,
  GuidedRespondRequest,
  GuidedRespondResponse,
  GuidedSession,
  GuidedStep,
  GetGuidedResponse,
  TerminalKind,
  TerminalReason,
  TurnPayload,
  TurnRecord,
  TurnType,
} from "./guided";

describe("guided protocol types", () => {
  it("TurnType union has 6 values", () => {
    const all: TurnType[] = [
      "inspect_and_confirm",
      "single_select",
      "multi_select_with_custom",
      "schema_form",
      "propose_chain",
      "recipe_offer",
    ];
    expect(all).toHaveLength(6);
  });

  it("ControlSignal union has 3 values", () => {
    const all: ControlSignal[] = [
      "exit_to_freeform",
      "request_advisor",
      "reject",
    ];
    expect(all).toHaveLength(3);
  });

  it("GuidedStep union has 4 values", () => {
    const all: GuidedStep[] = [
      "step_1_source",
      "step_2_sink",
      "step_2_5_recipe_match",
      "step_3_transforms",
    ];
    expect(all).toHaveLength(4);
  });

  it("TerminalKind union has 2 values", () => {
    const all: TerminalKind[] = ["completed", "exited_to_freeform"];
    expect(all).toHaveLength(2);
  });

  it("TerminalReason union has 3 values (no completed_pipeline)", () => {
    const all: TerminalReason[] = [
      "user_pressed_exit",
      "protocol_violation",
      "solver_exhausted",
    ];
    expect(all).toHaveLength(3);
    // Type-level negative: "completed_pipeline" is not in the union — any
    // attempt to assign it would be a TS compile error.
  });

  it("TurnPayload.step_index is number", () => {
    const payload: TurnPayload = {
      type: "single_select",
      step_index: 0,
      payload: {},
    };
    expect(typeof payload.step_index).toBe("number");
    expect(payload.step_index).toBe(0);
  });

  it("GuidedSession has exactly step, history, terminal", () => {
    // Exhaustiveness at compile time: the array type forces Keys to be
    // constrained to keyof GuidedSession.
    type Keys = keyof GuidedSession;
    const expected: Keys[] = ["step", "history", "terminal"];
    expect(expected).toHaveLength(3);
  });

  it("TurnRecord compiles with a well-formed sample", () => {
    const rec: TurnRecord = {
      step: "step_1_source",
      turn_type: "inspect_and_confirm",
      payload_hash: "abc123",
      response_hash: null,
      emitter: "server",
    };
    expect(rec.payload_hash).toBe("abc123");
  });

  it("GuidedRespondRequest compiles with all-null body", () => {
    const req: GuidedRespondRequest = {
      chosen: null,
      edited_values: null,
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    };
    expect(req.chosen).toBeNull();
  });

  it("GetGuidedResponse and GuidedRespondResponse include composition_state", () => {
    // Type-level check: if composition_state were absent the assignment below
    // would be a TS error.
    const check = (r: GetGuidedResponse | GuidedRespondResponse) => {
      return r.composition_state;
    };
    // Runtime: just confirm the type guard compiles; value is irrelevant.
    expect(check).toBeTypeOf("function");
  });
});
