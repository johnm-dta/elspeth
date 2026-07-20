import { describe, expect, it } from "vitest";

import { decodeGetGuidedResponse } from "./guidedDecoder";

function wireResponse(payloadOverrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    guided_session: {
      step: "step_4_wire",
      history: [{
        step: "step_4_wire",
        turn_type: "confirm_wiring",
        payload_hash: "a".repeat(64),
        response_hash: null,
        summary: null,
        emitter: "server",
      }],
      terminal: null,
      chat_history: [],
      chat_turn_seq: 0,
      profile: null,
    },
    next_turn: {
      type: "confirm_wiring",
      step_index: 3,
      turn_token: "b".repeat(64),
      payload: {
        proposal_id: "00000000-0000-4000-8000-000000000001",
        draft_hash: "d".repeat(64),
        sources: [],
        nodes: [],
        outputs: [],
        connections: [],
        semantic_contracts: [],
        warnings: [],
        blockers: [],
        can_confirm: true,
        ...payloadOverrides,
      },
    },
    terminal: null,
    composition_state: null,
  };
}

function aggregationNode(
  behaviorOverrides: Record<string, unknown> = {},
  cardinalityOverrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    stable_id: "00000000-0000-4000-8000-000000000002",
    label: "batch",
    node_type: "aggregation",
    plugin: "batch_stats",
    behavior: {
      kind: "aggregation",
      trigger_kinds: ["count", "timeout"],
      count: "25",
      timeout_seconds: 12.5,
      output_mode: "transform",
      expected_output_count: "1",
      ...behaviorOverrides,
    },
    required_fields: [],
    guaranteed_fields: [],
    row_cardinality: {
      input: "batch",
      output: "expected_count",
      expected_output_count: "1",
      ...cardinalityOverrides,
    },
    structured_output_fields: [],
  };
}

describe("guided schema-10 wire decoder", () => {
  it("decodes a wire turn bound to its pending proposal", () => {
    const decoded = decodeGetGuidedResponse(wireResponse());

    expect(decoded.next_turn?.type).toBe("confirm_wiring");
    if (decoded.next_turn?.type === "confirm_wiring") {
      expect(decoded.next_turn.payload.proposal_id).toBe("00000000-0000-4000-8000-000000000001");
      expect(decoded.next_turn.payload.draft_hash).toBe("d".repeat(64));
    }
  });

  it.each(["proposal_id", "draft_hash"])("rejects a wire turn missing %s", (missing) => {
    const response = wireResponse();
    const nextTurn = response.next_turn as { payload: Record<string, unknown> };
    delete nextTurn.payload[missing];

    expect(() => decodeGetGuidedResponse(response)).toThrow(missing);
  });

  it.each(["advisor_findings", "signoff_outcome", "passes_remaining"])(
    "rejects removed wire sign-off field %s",
    (field) => {
      expect(() => decodeGetGuidedResponse(wireResponse({ [field]: field === "passes_remaining" ? 1 : "legacy" })))
        .toThrow(`unexpected ${field}`);
    },
  );

  it.each([
    [["count", "unknown"], "unknown trigger"],
    [["count", "count"], "duplicate trigger"],
  ])("rejects %s aggregation trigger kinds", (triggerKinds) => {
    expect(() => decodeGetGuidedResponse(wireResponse({
      nodes: [aggregationNode({ trigger_kinds: triggerKinds })],
    }))).toThrow("trigger_kinds");
  });

  it.each([
    ["count trigger without count", { trigger_kinds: ["count", "timeout"], count: null }, "count"],
    ["count without count trigger", { trigger_kinds: ["timeout"], count: "25" }, "count"],
    ["timeout trigger without timeout", { trigger_kinds: ["count", "timeout"], timeout_seconds: null }, "timeout_seconds"],
    ["timeout without timeout trigger", { trigger_kinds: ["count"], timeout_seconds: 12.5 }, "timeout_seconds"],
  ])("rejects aggregation %s", (_case, behaviorOverrides, field) => {
    expect(() => decodeGetGuidedResponse(wireResponse({
      nodes: [aggregationNode(behaviorOverrides)],
    }))).toThrow(field);
  });

  it.each(["01", "+1", "1.0"])("rejects noncanonical cardinality count %s", (count) => {
    expect(() => decodeGetGuidedResponse(wireResponse({
      nodes: [aggregationNode({}, { expected_output_count: count })],
    }))).toThrow("expected_output_count");
  });

  it("rejects cardinality output/count coupling violations", () => {
    expect(() => decodeGetGuidedResponse(wireResponse({
      nodes: [aggregationNode({}, { output: "expected_count", expected_output_count: null })],
    }))).toThrow("expected_output_count");
    expect(() => decodeGetGuidedResponse(wireResponse({
      nodes: [aggregationNode({}, { output: "one", expected_output_count: "1" })],
    }))).toThrow("expected_output_count");
  });
});
