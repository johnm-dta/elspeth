import { describe, expect, it } from "vitest";
import { GUIDED_EXPLAIN_MESSAGE } from "./explainPrompt";
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
        { role: "user", content: "go", seq: 1, step: "step_1_source", ts_iso: "t", assistant_message_kind: null, synthetic_failure_reason: null },
        { role: "assistant", content: "Source created as a 3-row CSV.", seq: 2, step: "step_1_source", ts_iso: "t", assistant_message_kind: "assistant", synthetic_failure_reason: null },
        { role: "assistant", content: "Sink set.", seq: 4, step: "step_2_sink", ts_iso: "t", assistant_message_kind: "assistant", synthetic_failure_reason: null },
      ],
    });
    expect(latestAssistantRationale(s)).toBe("Source created as a 3-row CSV.");
  });

  it("returns null when no assistant turn exists for the step", () => {
    const s = session({
      step: "step_2_sink",
      chat_history: [{ role: "user", content: "go", seq: 1, step: "step_2_sink", ts_iso: "t", assistant_message_kind: null, synthetic_failure_reason: null }],
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
          assistant_message_kind: "assistant",
          synthetic_failure_reason: null,
        },
      ],
    });
    expect(latestAssistantRationale(s)).toBe("The source is built.");
  });

  it("rejects an over-long first line (falls back to the static step purpose)", () => {
    const s = session({
      chat_history: [
        { role: "assistant", content: "x".repeat(300), seq: 2, step: "step_1_source", ts_iso: "t", assistant_message_kind: "assistant", synthetic_failure_reason: null },
      ],
    });
    expect(latestAssistantRationale(s)).toBeNull();
  });

  it("unwraps inline markdown emphasis — the h2 renders plain text (operator-observed literal asterisks)", () => {
    const s = session({
      chat_history: [
        {
          role: "assistant",
          content: "You're at **Step 1: Source** — nothing configured `yet`.",
          seq: 2,
          step: "step_1_source",
          ts_iso: "t",
          assistant_message_kind: "assistant",
          synthetic_failure_reason: null,
        },
      ],
    });
    expect(latestAssistantRationale(s)).toBe(
      "You're at Step 1: Source — nothing configured yet.",
    );
  });

  it("skips replies to the Explain question — an explanation is not a build rationale", () => {
    const s = session({
      chat_history: [
        { role: "assistant", content: "Source created as a 3-row CSV.", seq: 2, step: "step_1_source", ts_iso: "t", assistant_message_kind: "assistant", synthetic_failure_reason: null },
        { role: "user", content: GUIDED_EXPLAIN_MESSAGE, seq: 3, step: "step_1_source", ts_iso: "t", assistant_message_kind: null, synthetic_failure_reason: null },
        { role: "assistant", content: "You're at Step 1 — here's everything set up so far.", seq: 4, step: "step_1_source", ts_iso: "t", assistant_message_kind: "assistant", synthetic_failure_reason: null },
      ],
    });
    // The headline stays the BUILD rationale (seq 2), not the higher-seq
    // explain answer that would otherwise hijack it.
    expect(latestAssistantRationale(s)).toBe("Source created as a 3-row CSV.");
  });

  it("falls back to the static purpose when the ONLY assistant turn is an explain reply", () => {
    const s = session({
      chat_history: [
        { role: "user", content: GUIDED_EXPLAIN_MESSAGE, seq: 1, step: "step_1_source", ts_iso: "t", assistant_message_kind: null, synthetic_failure_reason: null },
        { role: "assistant", content: "You're at Step 1 — nothing configured yet.", seq: 2, step: "step_1_source", ts_iso: "t", assistant_message_kind: "assistant", synthetic_failure_reason: null },
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
          assistant_message_kind: "assistant",
          synthetic_failure_reason: null,
        },
      ],
    });
    expect(latestAssistantRationale(s)).toBeNull();
  });

  it("skips a synthetic-failure turn — it must never become the decision headline (C-2)", () => {
    const s = session({
      chat_history: [
        {
          role: "assistant",
          content: "Source created as a 3-row CSV.",
          seq: 2,
          step: "step_1_source",
          ts_iso: "t",
          assistant_message_kind: "assistant",
          synthetic_failure_reason: null,
        },
        {
          role: "assistant",
          content: "I'm unavailable right now; you can still use the wizard controls.",
          seq: 3,
          step: "step_1_source",
          ts_iso: "t",
assistant_message_kind: "synthetic_failure",
          synthetic_failure_reason: "unavailable",
        },
      ],
    });
    // The higher-seq turn is synthetic — the real build rationale (seq 2)
    // stays the headline instead.
    expect(latestAssistantRationale(s)).toBe("Source created as a 3-row CSV.");
  });

  it("falls back to the static purpose when the ONLY assistant turn for the step is synthetic", () => {
    const s = session({
      step: "step_2_sink",
      chat_history: [
        { role: "user", content: "go", seq: 1, step: "step_2_sink", ts_iso: "t", assistant_message_kind: null, synthetic_failure_reason: null },
        {
          role: "assistant",
          content: "I'm unavailable right now; you can still use the wizard controls.",
          seq: 2,
          step: "step_2_sink",
          ts_iso: "t",
assistant_message_kind: "synthetic_failure",
          synthetic_failure_reason: "unavailable",
        },
      ],
    });
    expect(latestAssistantRationale(s)).toBeNull();
  });

  it("a stale synthetic-failure heading does not persist after the wizard advances", () => {
    // A synthetic failure at step_1 must not leak forward as the headline
    // once the wizard has moved on to step_2 — the per-step scoping (turn.step
    // !== session.step) already excludes it, independent of the kind check.
    const s = session({
      step: "step_2_sink",
      chat_history: [
        {
          role: "assistant",
          content: "I'm unavailable right now; you can still use the wizard controls.",
          seq: 1,
          step: "step_1_source",
          ts_iso: "t",
assistant_message_kind: "synthetic_failure",
          synthetic_failure_reason: "unavailable",
        },
      ],
    });
    expect(latestAssistantRationale(s)).toBeNull();
  });

  it("counts an explicitly classified assistant turn as real rationale", () => {
    const s = session({
      chat_history: [
        {
          role: "assistant",
          content: "Source created as a 3-row CSV.",
          seq: 2,
          step: "step_1_source",
          ts_iso: "t",
          assistant_message_kind: "assistant",
          synthetic_failure_reason: null,
        },
      ],
    });
    expect(latestAssistantRationale(s)).toBe("Source created as a 3-row CSV.");
  });
});
