import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { TutorialGuidedShell } from "./TutorialGuidedShell";
import { useSessionStore } from "@/stores/sessionStore";

const startGuidedSessionMock = vi.fn();
const startGuidedMock = vi.fn();

vi.mock("@/api/client", () => ({
  startGuidedSession: (...args: unknown[]) => startGuidedSessionMock(...args),
}));

vi.mock("@/components/chat/ChatPanel", () => ({
  ChatPanel: () => <div data-testid="chat-panel-stub" />,
}));

describe("TutorialGuidedShell", () => {
  beforeEach(() => {
    startGuidedSessionMock.mockReset().mockResolvedValue(undefined);
    startGuidedMock.mockReset().mockResolvedValue(undefined);
    // Start with NO active session so the test exercises the real production
    // path: TutorialGuidedShell must itself bind activeSessionId (D3/B4). A
    // pre-set activeSessionId here would mask a shell that forgot to bind it.
    useSessionStore.setState({
      activeSessionId: null,
      messages: [],
      compositionState: null,
      compositionProposals: [],
      composerPreferences: null,
      staleProposalIds: [],
      proposalActionPendingIds: [],
      composerProgress: null,
      stateVersions: [],
      isComposing: false,
      error: null,
      selectedNodeId: null,
      guidedSession: null,
      guidedNextTurn: null,
      guidedTerminal: null,
      guidedChatPending: false,
      guidedResponsePending: false,
      recoveryError: null,
      recoveryStartedCompositionVersion: null,
      startGuided: startGuidedMock,
    } as never);
  });

  it("posts the TUTORIAL profile and enters guided on mount", async () => {
    render(
      <TutorialGuidedShell sessionId="sess-1" onCompleted={vi.fn()} />,
    );
    await waitFor(() =>
      expect(startGuidedSessionMock).toHaveBeenCalledWith("sess-1", "tutorial"),
    );
    expect(startGuidedMock).toHaveBeenCalledWith("sess-1");
    // The shell must have bound the store's activeSessionId; otherwise
    // startGuided discards its payload and ChatPanel renders the empty surface.
    expect(useSessionStore.getState().activeSessionId).toBe("sess-1");
  });

  it("renders the real ChatPanel guided surface", async () => {
    render(
      <TutorialGuidedShell sessionId="sess-1" onCompleted={vi.fn()} />,
    );
    await waitFor(() =>
      expect(screen.getByTestId("chat-panel-stub")).toBeInTheDocument(),
    );
  });

  it("clears stale completed guided state before starting a new tutorial session", async () => {
    const onCompleted = vi.fn();
    useSessionStore.setState({
      activeSessionId: "old-sess",
      messages: [{ id: "old-message" }],
      compositionState: { id: "old-state", version: 99 },
      compositionProposals: [{ id: "old-proposal" }],
      guidedSession: {
        step: "step_4_wire",
        history: [],
        terminal: { kind: "completed", reason: null },
        chat_history: [],
        chat_turn_seq: 0,
        profile: null,
      },
      guidedNextTurn: null,
      guidedTerminal: { kind: "completed", reason: null },
    } as never);
    render(
      <TutorialGuidedShell sessionId="sess-2" onCompleted={onCompleted} />,
    );

    await waitFor(() =>
      expect(startGuidedSessionMock).toHaveBeenCalledWith("sess-2", "tutorial"),
    );
    expect(useSessionStore.getState().activeSessionId).toBe("sess-2");
    expect(useSessionStore.getState().guidedSession).toBeNull();
    expect(useSessionStore.getState().guidedNextTurn).toBeNull();
    expect(useSessionStore.getState().guidedTerminal).toBeNull();
    expect(useSessionStore.getState().messages).toEqual([]);
    expect(useSessionStore.getState().compositionState).toBeNull();
    expect(onCompleted).not.toHaveBeenCalled();
  });

  it("calls onCompleted when the guided session terminal is completed", async () => {
    const onCompleted = vi.fn();
    render(
      <TutorialGuidedShell sessionId="sess-1" onCompleted={onCompleted} />,
    );
    await waitFor(() => expect(startGuidedMock).toHaveBeenCalled());
    useSessionStore.setState({
      guidedSession: {
        step: "step_4_wire",
        history: [],
        terminal: { kind: "completed", reason: null },
        chat_history: [],
        chat_turn_seq: 0,
        profile: null,
      },
    } as never);
    await waitFor(() => expect(onCompleted).toHaveBeenCalledWith("sess-1"));
  });

  it("shows a user-visible error if guided startup fails", async () => {
    startGuidedSessionMock.mockRejectedValueOnce(new Error("start failed"));
    render(
      <TutorialGuidedShell sessionId="sess-1" onCompleted={vi.fn()} />,
    );
    expect(screen.getByRole("status")).toHaveTextContent("Starting guided composer");
    expect(await screen.findByRole("alert")).toHaveTextContent("start failed");
    expect(startGuidedMock).not.toHaveBeenCalled();
  });

  it("shows a user-visible error if store guided entry fails", async () => {
    startGuidedMock.mockRejectedValueOnce(new Error("store failed"));
    render(
      <TutorialGuidedShell sessionId="sess-1" onCompleted={vi.fn()} />,
    );
    expect(await screen.findByRole("alert")).toHaveTextContent("store failed");
  });
});
