// ============================================================================
// isGuidedBuildActive — the shared "guided build on screen" predicate.
//
// Two consumers must agree exactly: ChatPanel renders the two-column guided
// workspace when this holds, and App suppresses the freeform SideRail for
// the same states (the workspace rail replaces it). These tests pin the
// branch table so a future edit to either consumer's expectations has to
// come through here.
// ============================================================================

import { describe, expect, it } from "vitest";
import { isGuidedBuildActive } from "./guidedBuildActive";
import type { GuidedSession, TurnPayload } from "@/types/guided";

function session(overrides: Partial<GuidedSession> = {}): GuidedSession {
  return {
    step: "step_1_source",
    history: [],
    terminal: null,
    chat_history: [],
    chat_turn_seq: 0,
    profile: null,
    ...overrides,
  };
}

const TURN = {
  type: "single_select",
  step_index: 0,
} as unknown as TurnPayload;

describe("isGuidedBuildActive", () => {
  it("is false with no guided session (freeform keeps its SideRail)", () => {
    expect(isGuidedBuildActive(null, null)).toBe(false);
  });

  it("is true for a non-terminal session with a posed turn", () => {
    expect(isGuidedBuildActive(session(), TURN)).toBe(true);
  });

  it("is true at step_3_transforms with NO turn (chat-driven cold start still needs the composer)", () => {
    expect(
      isGuidedBuildActive(session({ step: "step_3_transforms" }), null),
    ).toBe(true);
  });

  it("is false for a turn-less non-terminal state at other steps (ChatPanel falls through to freeform, rail included)", () => {
    expect(isGuidedBuildActive(session(), null)).toBe(false);
  });

  it("is false once completed — the SideRail must return (Run/Export live there post-completion)", () => {
    expect(
      isGuidedBuildActive(
        session({
          step: "step_4_wire",
          terminal: { kind: "completed", reason: null, pipeline_yaml: "p: {}" },
        }),
        TURN,
      ),
    ).toBe(false);
  });

  it("is false after exit-to-freeform — the freeform surface gets its rail back", () => {
    expect(
      isGuidedBuildActive(
        session({
          terminal: {
            kind: "exited_to_freeform",
            reason: "user_pressed_exit",
            pipeline_yaml: null,
          },
        }),
        null,
      ),
    ).toBe(false);
  });
});
