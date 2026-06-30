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
  WireStageData,
  WorkflowProfile,
} from "./guided";

// Compile-time mutual-extends check.
type Equals<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false;

describe("guided protocol types", () => {
  it("TurnType union has exactly 7 values", () => {
    const _exact: Equals<
      TurnType,
      | "inspect_and_confirm"
      | "single_select"
      | "multi_select_with_custom"
      | "schema_form"
      | "propose_chain"
      | "recipe_offer"
      | "confirm_wiring"
    > = true;
    const all: TurnType[] = [
      "inspect_and_confirm",
      "single_select",
      "multi_select_with_custom",
      "schema_form",
      "propose_chain",
      "recipe_offer",
      "confirm_wiring",
    ];
    expect(_exact).toBe(true);
    expect(all).toHaveLength(7);
  });

  it("ControlSignal union has 3 values", () => {
    const _exact: Equals<
      ControlSignal,
      "exit_to_freeform" | "request_advisor" | "reject"
    > = true;
    const all: ControlSignal[] = [
      "exit_to_freeform",
      "request_advisor",
      "reject",
    ];
    expect(_exact).toBe(true);
    expect(all).toHaveLength(3);
  });

  it("GuidedStep union has exactly 5 values", () => {
    const _exact: Equals<
      GuidedStep,
      | "step_1_source"
      | "step_2_sink"
      | "step_2_5_recipe_match"
      | "step_3_transforms"
      | "step_4_wire"
    > = true;
    const all: GuidedStep[] = [
      "step_1_source",
      "step_2_sink",
      "step_2_5_recipe_match",
      "step_3_transforms",
      "step_4_wire",
    ];
    expect(_exact).toBe(true);
    expect(all).toHaveLength(5);
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

  it("TurnPayload.step_index is number (compile-time) and 0-based ordinal", () => {
    const payload: TurnPayload = {
      type: "single_select",
      step_index: 0,
      payload: {},
    };
    expect(payload.step_index).toBe(0);
  });

  it("GuidedSession has exactly step, history, terminal, chat_history, chat_turn_seq, profile — exhaustive", () => {
    // Compile-time mutual-extends: adding/removing a key in GuidedSession
    // makes this assignment fail tsc.  Slice 5 added chat_history and
    // chat_turn_seq; P6.2 added profile (server-owned WorkflowProfile).
    const _exact: Equals<
      keyof GuidedSession,
      | "step"
      | "history"
      | "terminal"
      | "chat_history"
      | "chat_turn_seq"
      | "profile"
    > = true;
    expect(_exact).toBe(true);
  });

  it("TurnRecord nullable response_hash is honoured", () => {
    const rec: TurnRecord = {
      step: "step_1_source",
      turn_type: "inspect_and_confirm",
      payload_hash: "abc123",
      response_hash: null,
      summary: null,
      emitter: "server",
    };
    expect(rec.response_hash).toBeNull();
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

describe("WorkflowProfile wire type", () => {
  it("carries exactly the four wire-visible boolean flags", () => {
    // Compile-time exhaustive check: the wire profile is exactly these four
    // behavior flags — adding any other key here would fail tsc.
    const _exact: Equals<
      keyof WorkflowProfile,
      "coaching" | "bookends" | "recipe_match" | "advisor_checkpoints"
    > = true;
    const profile: WorkflowProfile = {
      coaching: true,
      bookends: true,
      recipe_match: true,
      advisor_checkpoints: true,
    };
    expect(_exact).toBe(true);
    expect(profile.advisor_checkpoints).toBe(true);
  });

  it("rides GuidedSession.profile; null is the empty/live-guided profile", () => {
    const profile: WorkflowProfile = {
      coaching: false,
      bookends: false,
      recipe_match: false,
      advisor_checkpoints: false,
    };
    const seeded: Pick<GuidedSession, "profile"> = { profile };
    expect(seeded.profile).not.toBeNull();
    const empty: Pick<GuidedSession, "profile"> = { profile: null };
    expect(empty.profile).toBeNull();
  });
});

describe("WireStageData wire shape", () => {
  it("uses topology ids and edge_contracts from/to keys", () => {
    const data: WireStageData = {
      topology: {
        sources: {
          source: {
            id: "source",
            plugin: "inline_blob",
            on_success: "chain_in",
            on_validation_failure: "discard",
          },
        },
        nodes: [
          {
            id: "scrape",
            node_type: "transform",
            plugin: "web_scrape",
            input: "chain_in",
            on_success: "scraped",
            on_error: "scrape_error",
            routes: { retry: "chain_in" },
            fork_to: ["audit_stream"],
            branches: null,
          },
        ],
        outputs: [
          {
            id: "output:jsonl_out",
            sink_name: "jsonl_out",
            plugin: "json",
            on_write_failure: "discard",
          },
        ],
      },
      edge_contracts: [
        {
          from: "scrape",
          to: "output:jsonl_out",
          producer_guarantees: ["url", "body"],
          consumer_requires: ["body"],
          missing_fields: [],
          satisfied: true,
        },
      ],
      semantic_contracts: [],
      warnings: [],
      advisor_findings: "Prompt shield warning reviewed.",
      signoff_outcome: "approved",
    };

    expect(data.edge_contracts[0].from).toBe("scrape");
    expect(data.edge_contracts[0].to).toBe("output:jsonl_out");
    // @ts-expect-error edge_contracts keys are from/to, NOT from_id.
    expect(data.edge_contracts[0].from_id).toBeUndefined();
  });
});
