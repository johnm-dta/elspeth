import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CommandPalette } from "./CommandPalette";
import { useSessionStore } from "@/stores/sessionStore";
import { resetStore } from "@/test/store-helpers";
import type { GuidedSession } from "@/types/guided";

vi.mock("@/api/client", () => ({
  fetchSessions: vi.fn(),
  createSession: vi.fn(),
  fetchMessages: vi.fn(),
  fetchCompositionState: vi.fn(),
  fetchComposerProgress: vi.fn(),
  sendMessage: vi.fn(),
  recompose: vi.fn(),
  forkFromMessage: vi.fn(),
  revertToVersion: vi.fn(),
  fetchStateVersions: vi.fn(),
  archiveSession: vi.fn(),
  getGuided: vi.fn(),
  respondGuided: vi.fn(),
  reenterGuided: vi.fn(),
  chatGuided: vi.fn(),
}));

vi.mock("@/stores/executionStore", () => ({
  useExecutionStore: (selector: (state: unknown) => unknown) =>
    selector({
      validate: vi.fn(),
      execute: vi.fn(),
      validationResult: null,
    }),
}));

const exitedGuidedSession: GuidedSession = {
  step: "step_1_source",
  history: [],
  terminal: {
    kind: "exited_to_freeform",
    reason: "user_pressed_exit",
    pipeline_yaml: null,
  },
  chat_history: [],
  chat_turn_seq: 0,
};

describe("CommandPalette guided-mode commands", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    Element.prototype.scrollIntoView = vi.fn();
    resetStore(useSessionStore);
  });

  it("offers Re-enter guided mode for a user-exited guided session", async () => {
    const user = userEvent.setup();
    const reenterGuided = vi.fn().mockResolvedValue(undefined);
    const onClose = vi.fn();
    useSessionStore.setState({
      activeSessionId: "session-1",
      guidedSession: exitedGuidedSession,
      guidedTerminal: exitedGuidedSession.terminal,
      reenterGuided,
    });

    render(<CommandPalette isOpen onClose={onClose} />);

    await user.click(
      screen.getByRole("option", { name: /re-enter guided mode/i }),
    );

    expect(reenterGuided).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("does not offer Re-enter guided mode for solver auto-drops", () => {
    useSessionStore.setState({
      activeSessionId: "session-1",
      guidedSession: {
        ...exitedGuidedSession,
        terminal: {
          kind: "exited_to_freeform",
          reason: "solver_exhausted",
          pipeline_yaml: null,
        },
      },
    });

    render(<CommandPalette isOpen onClose={vi.fn()} />);

    expect(
      screen.queryByRole("option", { name: /re-enter guided mode/i }),
    ).toBeNull();
  });

  it("does not offer navigation to the removed Runs tab", () => {
    render(<CommandPalette isOpen onClose={vi.fn()} />);

    expect(
      screen.queryByRole("option", { name: /Switch to Runs Tab/i }),
    ).toBeNull();
  });
});
