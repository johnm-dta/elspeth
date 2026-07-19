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
        topology: { sources: {}, nodes: [], outputs: [] },
        edge_contracts: [],
        semantic_contracts: [],
        warnings: [],
        ...payloadOverrides,
      },
    },
    terminal: null,
    composition_state: null,
  };
}

describe("guided schema-9 decoder", () => {
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
});
