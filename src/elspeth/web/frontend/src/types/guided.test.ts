// Tests for guided.ts — type-assertion style.
// Vitest compile failures ARE the test: if the union drifts, the array literal
// above it stops compiling.

import { describe, expect, it } from "vitest";

import type {
  ChatTurn,
  ControlSignal,
  GuidedChatResponse,
  GuidedRespondAction,
  GuidedRespondRequest,
  GuidedRespondResponse,
  GuidedSession,
  GuidedStep,
  GetGuidedResponse,
  TerminalKind,
  TerminalState,
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
      | "review_components"
      | "propose_pipeline"
      | "confirm_wiring"
    > = true;
    const all: TurnType[] = [
      "inspect_and_confirm",
      "single_select",
      "multi_select_with_custom",
      "schema_form",
      "review_components",
      "propose_pipeline",
      "confirm_wiring",
    ];
    expect(_exact).toBe(true);
    expect(all).toHaveLength(7);
  });

  it("ControlSignal union has 5 values (C-3: added back + passthrough, fixing pre-existing drift from protocol.py)", () => {
    const _exact: Equals<
      ControlSignal,
      "exit_to_freeform" | "request_advisor" | "reject" | "back" | "passthrough"
    > = true;
    const all: ControlSignal[] = [
      "exit_to_freeform",
      "request_advisor",
      "reject",
      "back",
      "passthrough",
    ];
    expect(_exact).toBe(true);
    expect(all).toHaveLength(5);
  });

  it("GuidedStep union has exactly 4 values", () => {
    const _exact: Equals<
      GuidedStep,
      | "step_1_source"
      | "step_2_sink"
      | "step_3_transforms"
      | "step_4_wire"
    > = true;
    const all: GuidedStep[] = [
      "step_1_source",
      "step_2_sink",
      "step_3_transforms",
      "step_4_wire",
    ];
    expect(_exact).toBe(true);
    expect(all).toHaveLength(4);
  });

  it("TerminalKind union has 2 values", () => {
    const all: TerminalKind[] = ["completed", "exited_to_freeform"];
    expect(all).toHaveLength(2);
  });

  it("TerminalReason has only the reversible user exit", () => {
    const all: TerminalReason[] = ["user_pressed_exit"];
    expect(all).toHaveLength(1);
    // Type-level negative: "completed_pipeline" is not in the union — any
    // attempt to assign it would be a TS compile error.
  });

  it("TurnPayload is discriminated by an exact type/payload pair", () => {
    const payload: TurnPayload = {
      type: "single_select",
      step_index: 0,
      turn_token: "a".repeat(64),
      payload: { question: "Choose", options: [], allow_custom: false },
    };
    expect(payload.step_index).toBe(0);

    const mismatched: TurnPayload = {
      type: "single_select",
      step_index: 0,
      turn_token: "a".repeat(64),
      // @ts-expect-error inspect payloads cannot ride the single-select discriminator
      payload: { observed: { columns: [], samples: [], warnings: [] } },
    };
    expect(mismatched.type).toBe("single_select");
  });

  it("wire review remains bound to the pending proposal", () => {
    const payload: WireStageData = {
      proposal_id: "00000000-0000-4000-8000-000000000001",
      draft_hash: "d".repeat(64),
      topology: { sources: {}, nodes: [], outputs: [] },
      edge_contracts: [],
      semantic_contracts: [],
      warnings: [],
    };
    expect(payload.proposal_id).toMatch(/^[0-9a-f-]{36}$/);
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

  it("GuidedRespondRequest rejects an all-null live action", () => {
    // @ts-expect-error a live request must name one closed legal action
    const invalid: GuidedRespondRequest = {
      operation_id: "00000000-0000-4000-8000-000000000601",
      turn_token: "a".repeat(64),
      chosen: null,
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
    };
    expect(invalid.chosen).toBeNull();
  });

  it("GuidedRespondAction rejects empty selection and custom semantic no-ops", () => {
    const emptyChosen: GuidedRespondAction = {
      // @ts-expect-error an empty chosen action without non-empty custom input is not actionable
      chosen: [],
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
    };
    const emptyCustom: GuidedRespondAction = {
      chosen: null,
      edited_values: null,
      // @ts-expect-error an empty custom-only action is not actionable
      custom_inputs: [],
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
    };
    expect(emptyChosen.chosen).toEqual([]);
    expect(emptyCustom.custom_inputs).toEqual([]);
  });

  it("GuidedRespondAction uses a target-only proposal revision contract", () => {
    const targetOnlyRevision: GuidedRespondAction = {
      chosen: null,
      edited_values: null,
      custom_inputs: null,
      proposal_id: "00000000-0000-4000-8000-000000000701",
      draft_hash: "f".repeat(64),
      edit_target: {
        kind: "node",
        stable_id: "00000000-0000-4000-8000-000000000702",
      },
      control_signal: null,
    };
    // @ts-expect-error proposal revisions name only the durable target; inline edited values are not accepted
    const revisionWithInlineValues: GuidedRespondAction = {
      ...targetOnlyRevision,
      edited_values: {},
    };
    expect(targetOnlyRevision.edit_target?.kind).toBe("node");
    expect(revisionWithInlineValues.edited_values).toEqual({});
  });

  it("GuidedRespondAction closes unrelated fields for every exact response action", () => {
    // @ts-expect-error chosen and edited_values are mutually exclusive actions
    const chosenAndEdited: GuidedRespondAction = {
      chosen: ["csv"],
      edited_values: { plugin: "csv", options: {} },
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
    };
    const chosenAndCustom: GuidedRespondAction = {
      chosen: ["name"],
      edited_values: null,
      custom_inputs: ["extra"],
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
    };
    // @ts-expect-error edited_values cannot be combined with custom_inputs
    const editedAndCustom: GuidedRespondAction = {
      chosen: null,
      edited_values: { plugin: "csv", options: {} },
      custom_inputs: ["extra"],
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
    };
    expect(chosenAndEdited.chosen).toEqual(["csv"]);
    expect(chosenAndCustom.custom_inputs).toEqual(["extra"]);
    expect(editedAndCustom.edited_values).toEqual({ plugin: "csv", options: {} });
  });

  it("TerminalState is discriminated by kind and its cross-fields", () => {
    const completed: TerminalState = {
      kind: "completed",
      reason: null,
      pipeline_yaml: "sources: {}",
    };
    const exited: TerminalState = {
      kind: "exited_to_freeform",
      reason: "user_pressed_exit",
      pipeline_yaml: null,
    };
    const completedWithReason: TerminalState = {
      kind: "completed",
      // @ts-expect-error completed terminals never carry an exit reason
      reason: "user_pressed_exit",
      pipeline_yaml: "sources: {}",
    };
    // @ts-expect-error exited terminals never carry pipeline YAML
    const exitedWithYaml: TerminalState = {
      kind: "exited_to_freeform",
      reason: "user_pressed_exit",
      pipeline_yaml: "sources: {}",
    };
    expect(completed.kind).toBe("completed");
    expect(exited.kind).toBe("exited_to_freeform");
    expect(completedWithReason.reason).toBe("user_pressed_exit");
    expect(exitedWithYaml.pipeline_yaml).toBe("sources: {}");
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

  it("ChatTurn carries exact required assistant and synthetic-failure discriminators", () => {
    const withKind: ChatTurn = {
      role: "assistant",
      content: "I'm unavailable right now; you can still use the wizard controls.",
      seq: 1,
      step: "step_1_source",
      ts_iso: "t",
      assistant_message_kind: "synthetic_failure",
      synthetic_failure_reason: "unavailable",
    };
    expect(withKind.assistant_message_kind).toBe("synthetic_failure");
    expect(withKind.synthetic_failure_reason).toBe("unavailable");

    const notApplied: ChatTurn = {
      role: "assistant",
      content: "I did not apply generated source content.",
      seq: 3,
      step: "step_1_source",
      ts_iso: "t",
      assistant_message_kind: "synthetic_failure",
      synthetic_failure_reason: "not_applied",
    };
    expect(notApplied.synthetic_failure_reason).toBe("not_applied");
  });

  it("GuidedChatResponse.assistant_message_kind is required and typed to the same two values", () => {
    const response: GuidedChatResponse = {
      assistant_message: "I'm unavailable right now; you can still use the wizard controls.",
      assistant_message_kind: "synthetic_failure",
      guided_session: {
        step: "step_1_source",
        history: [],
        terminal: null,
        chat_history: [],
        chat_turn_seq: 0,
        profile: null,
      },
      next_turn: {
        type: "single_select",
        step_index: 0,
        turn_token: "a".repeat(64),
        payload: { question: "Choose", options: [], allow_custom: false },
      },
      terminal: null,
      composition_state: {
        id: "state-1",
        session_id: "session-1",
        version: 1,
        nodes: [],
        edges: [],
        sources: {},
        outputs: [],
        metadata: { name: null, description: null },
        is_valid: true,
        validation_errors: null,
        validation_warnings: null,
        validation_suggestions: null,
        derived_from_state_id: null,
        created_at: "2026-07-19T00:00:00Z",
        composer_meta: null,
        plugin_policy_findings: [],
      },
    };
    expect(response.assistant_message_kind).toBe("synthetic_failure");
  });
});

describe("WorkflowProfile wire type", () => {
  it("carries exactly the three wire-visible boolean flags", () => {
    // Compile-time exhaustive check: the wire profile is exactly these three
    // behavior flags — adding any other key here would fail tsc.
    const _exact: Equals<
      keyof WorkflowProfile,
      "coaching" | "bookends" | "advisor_checkpoints"
    > = true;
    const profile: WorkflowProfile = {
      coaching: true,
      bookends: true,
      advisor_checkpoints: true,
    };
    expect(_exact).toBe(true);
    expect(profile.advisor_checkpoints).toBe(true);
  });

  it("rides GuidedSession.profile; null is the empty/live-guided profile", () => {
    const profile: WorkflowProfile = {
      coaching: false,
      bookends: false,
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
      proposal_id: "00000000-0000-4000-8000-000000000001",
      draft_hash: "d".repeat(64),
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
