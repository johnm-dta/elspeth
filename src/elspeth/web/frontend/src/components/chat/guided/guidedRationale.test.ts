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
});
