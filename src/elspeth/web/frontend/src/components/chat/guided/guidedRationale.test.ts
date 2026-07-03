import { describe, expect, it } from "vitest";
import { latestAssistantRationale } from "./guidedRationale";
import type { GuidedSession } from "@/types/guided";

function session(overrides: Partial<GuidedSession>): GuidedSession {
  return {
    step: "step_1_source",
    history: [],
    terminal: null,
    chat_history: [],
    chat_turn_seq: 0,
    profile: null,
    ...overrides,
  } as GuidedSession;
}

describe("latestAssistantRationale", () => {
  it("returns the highest-seq assistant turn for the current step", () => {
    const s = session({
      step: "step_1_source",
      chat_history: [
        { role: "user", content: "go", seq: 1, step: "step_1_source", ts_iso: "t" },
        { role: "assistant", content: "Source created as a 3-row CSV.", seq: 2, step: "step_1_source", ts_iso: "t" },
        { role: "assistant", content: "Sink set.", seq: 4, step: "step_2_sink", ts_iso: "t" },
      ],
    });
    expect(latestAssistantRationale(s)).toBe("Source created as a 3-row CSV.");
  });

  it("returns null when no assistant turn exists for the step", () => {
    const s = session({
      step: "step_2_sink",
      chat_history: [{ role: "user", content: "go", seq: 1, step: "step_2_sink", ts_iso: "t" }],
    });
    expect(latestAssistantRationale(s)).toBeNull();
  });

  it("uses only the FIRST line of a multi-paragraph reply (the full reply lives in the bubble)", () => {
    const s = session({
      chat_history: [
        {
          role: "assistant",
          content: "The source is built.\n\nHere's what was configured and why:\n| Decision | Reasoning |",
          seq: 2,
          step: "step_1_source",
          ts_iso: "t",
        },
      ],
    });
    expect(latestAssistantRationale(s)).toBe("The source is built.");
  });

  it("rejects an over-long first line (falls back to the static step purpose)", () => {
    const s = session({
      chat_history: [
        { role: "assistant", content: "x".repeat(300), seq: 2, step: "step_1_source", ts_iso: "t" },
      ],
    });
    expect(latestAssistantRationale(s)).toBeNull();
  });

  it("rejects tool-call scaffolding leaked into the message (renders as a wall of raw markup otherwise)", () => {
    const s = session({
      chat_history: [
        {
          role: "assistant",
          content: '<tool_call>{"name": "list_sources"}</tool_call> then some prose',
          seq: 2,
          step: "step_1_source",
          ts_iso: "t",
        },
      ],
    });
    expect(latestAssistantRationale(s)).toBeNull();
  });
});
