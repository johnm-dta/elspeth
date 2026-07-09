import { StrictMode } from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { TutorialGuidedShell } from "./TutorialGuidedShell";
import {
  TUTORIAL_SINK_PROMPT,
  TUTORIAL_SOURCE_PROMPT,
  TUTORIAL_TRANSFORMS_PROMPT,
} from "./tutorialMachine";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useSessionStore } from "@/stores/sessionStore";

const startGuidedSessionMock = vi.fn();
const startGuidedMock = vi.fn();
const getTutorialSampleMock = vi.fn();
const respondGuidedMock = vi.fn();
const refreshInterpretationsMock = vi.fn();

const SAMPLE_URLS = [
  "https://elspeth.example/tutorial-site/project-1.html",
  "https://elspeth.example/tutorial-site/project-2.html",
  "https://elspeth.example/tutorial-site/project-3.html",
];

type TerminalKind = "completed" | "exited_to_freeform";

// Build a minimal GuidedSession payload for the store. `terminalKind=null`
// yields a live, not-yet-terminal wizard (the state TutorialGuidedShell must
// OBSERVE before a later completion may graduate).
function guidedSessionPayload(terminalKind: TerminalKind | null): unknown {
  return {
    step: "step_4_wire",
    history: [],
    terminal: terminalKind === null ? null : { kind: terminalKind, reason: null },
    chat_history: [],
    chat_turn_seq: 0,
    profile: null,
  };
}

vi.mock("@/api/client", () => ({
  startGuidedSession: (...args: unknown[]) => startGuidedSessionMock(...args),
  getTutorialSample: (...args: unknown[]) => getTutorialSampleMock(...args),
  respondGuided: (...args: unknown[]) => respondGuidedMock(...args),
}));

vi.mock("@/components/chat/ChatPanel", () => ({
  ChatPanel: (props: {
    isTutorial?: boolean;
    lockedChatPrompt?: Partial<Record<string, string>>;
  }) => (
    <div
      data-testid="chat-panel-stub"
      data-is-tutorial={String(props.isTutorial)}
      data-locked-prompt={props.lockedChatPrompt?.step_1_source}
      data-locked-sink={props.lockedChatPrompt?.step_2_sink}
      data-locked-transforms={props.lockedChatPrompt?.step_3_transforms}
    />
  ),
}));

describe("TutorialGuidedShell", () => {
  beforeEach(() => {
    startGuidedSessionMock.mockReset().mockResolvedValue(undefined);
    startGuidedMock.mockReset().mockResolvedValue(undefined);
    getTutorialSampleMock
      .mockReset()
      .mockResolvedValue({ sample_urls: SAMPLE_URLS });
    respondGuidedMock.mockReset().mockResolvedValue({
      guided_session: guidedSessionPayload("exited_to_freeform"),
      next_turn: null,
      terminal: { kind: "exited_to_freeform", reason: "user_pressed_exit" },
      composition_state: null,
    });
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
    // The shell rehydrates the interpretation-event projection on start (the
    // tutorial bridge bypasses selectSession, which normally does this) — stub
    // the store action so the mocked api module doesn't need the route.
    refreshInterpretationsMock.mockReset().mockResolvedValue(undefined);
    useInterpretationEventsStore.setState({
      refreshAll: refreshInterpretationsMock,
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

  it("routes a 404 start failure to onSessionMissing instead of the dead-end error", async () => {
    // The persisted resume session can be gone (swept/archived/wiped). The
    // shell must hand recovery to the parent — rendering the raw error here
    // strands the learner: no ChatPanel, no skip, no forward affordance.
    startGuidedSessionMock
      .mockReset()
      .mockRejectedValue({ status: 404, detail: "Session not found" });
    const onSessionMissing = vi.fn();
    render(
      <TutorialGuidedShell
        sessionId="sess-gone"
        onCompleted={vi.fn()}
        onSessionMissing={onSessionMissing}
      />,
    );
    await waitFor(() => expect(onSessionMissing).toHaveBeenCalledTimes(1));
    // The dead id is passed so the parent can key idempotency on it and
    // release the store binding for exactly that session.
    expect(onSessionMissing).toHaveBeenCalledWith("sess-gone");
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("rehydrates pending interpretation events on start (resume must not drop the ack gate)", async () => {
    // The tutorial bridge bypasses selectSession (which normally rehydrates
    // pendingBySession). Without this call a mid-Build reload resumed with NO
    // pending acknowledgement cards: the wire-stage Confirm was not blocked
    // and the run later failed server-side with
    // UnresolvedInterpretationPlaceholderError.
    render(
      <TutorialGuidedShell sessionId="sess-1" onCompleted={vi.fn()} />,
    );
    await waitFor(() =>
      expect(refreshInterpretationsMock).toHaveBeenCalledWith("sess-1"),
    );
  });

  it("mounts the ChatPanel guided surface with isTutorial set", async () => {
    // ChatPanel is stubbed at the module boundary (see vi.mock above). The
    // tutorial shell MUST pass isTutorial so ChatPanel suppresses the
    // freeform exits and never falls through to the freeform body (concern B).
    render(
      <TutorialGuidedShell sessionId="sess-1" onCompleted={vi.fn()} />,
    );
    const stub = await screen.findByTestId("chat-panel-stub");
    expect(stub).toBeInTheDocument();
    expect(stub.dataset.isTutorial).toBe("true");
  });

  it("gives each stage its own prompt; appends sample URLs to the SOURCE stage only", async () => {
    // Per-stage staged orchestrator: the source prompt names no URLs in its
    // constant; the shell appends the runtime-resolved synthetic URLs (8a GET
    // surface) to the SOURCE stage only. Sink/transforms get their own focused
    // prompts with no URLs.
    render(<TutorialGuidedShell sessionId="sess-1" onCompleted={vi.fn()} />);
    await waitFor(() =>
      expect(getTutorialSampleMock).toHaveBeenCalledWith("sess-1"),
    );
    const stub = await screen.findByTestId("chat-panel-stub");
    const lockedSource = stub.dataset.lockedPrompt ?? "";
    expect(lockedSource).toContain(TUTORIAL_SOURCE_PROMPT);
    for (const url of SAMPLE_URLS) {
      expect(lockedSource).toContain(url);
    }
    // URLs come AFTER the source prose (appended, not interleaved).
    expect(lockedSource.indexOf(SAMPLE_URLS[0])).toBeGreaterThan(
      lockedSource.indexOf(TUTORIAL_SOURCE_PROMPT),
    );
    // Sink and transforms stages carry their own focused prompts, no URLs.
    expect(stub.dataset.lockedSink).toBe(TUTORIAL_SINK_PROMPT);
    expect(stub.dataset.lockedTransforms).toBe(TUTORIAL_TRANSFORMS_PROMPT);
    expect(stub.dataset.lockedTransforms).not.toContain(SAMPLE_URLS[0]);
  });

  it("gates the chat panel until the sample URLs resolve (never an editable box)", async () => {
    // Hold the sample fetch open so we can observe the pre-resolve state: the
    // ChatPanel (and thus the locked input) must NOT render yet — a loading
    // status stands in. This is the correctness gate: the learner can never
    // Send the URL-less canonical prompt.
    let resolveSample: (value: unknown) => void = () => undefined;
    getTutorialSampleMock.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveSample = resolve;
      }),
    );
    render(<TutorialGuidedShell sessionId="sess-1" onCompleted={vi.fn()} />);
    await waitFor(() =>
      expect(getTutorialSampleMock).toHaveBeenCalledWith("sess-1"),
    );
    // Pre-resolve: no chat panel, loading status visible.
    expect(screen.queryByTestId("chat-panel-stub")).not.toBeInTheDocument();
    expect(
      screen.getByText("Preparing the tutorial's sample pages…"),
    ).toBeInTheDocument();
    // Resolve and the panel appears with the URL-bearing locked prompt.
    resolveSample({ sample_urls: SAMPLE_URLS });
    const stub = await screen.findByTestId("chat-panel-stub");
    expect(stub.dataset.lockedPrompt).toContain(SAMPLE_URLS[2]);
  });

  it("exits server-side and stops startup when exit is requested before guided state loads", async () => {
    let resolveStart: (value: unknown) => void = () => undefined;
    const exitRequestedRef = { current: false };
    startGuidedSessionMock.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveStart = resolve;
      }),
    );
    render(
      <TutorialGuidedShell
        sessionId="sess-1"
        onCompleted={vi.fn()}
        exitRequestedRef={exitRequestedRef}
      />,
    );
    await waitFor(() =>
      expect(startGuidedSessionMock).toHaveBeenCalledWith("sess-1", "tutorial"),
    );

    exitRequestedRef.current = true;
    resolveStart({});

    await waitFor(() =>
      expect(respondGuidedMock).toHaveBeenCalledWith("sess-1", {
        chosen: null,
        edited_values: null,
        custom_inputs: null,
        accepted_step_index: null,
        edit_step_index: null,
        control_signal: "exit_to_freeform",
      }),
    );
    expect(getTutorialSampleMock).not.toHaveBeenCalled();
    expect(startGuidedMock).not.toHaveBeenCalled();
    expect(screen.queryByTestId("chat-panel-stub")).not.toBeInTheDocument();
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

  it("calls onCompleted when the guided session transitions to completed", async () => {
    const onCompleted = vi.fn();
    render(
      <TutorialGuidedShell sessionId="sess-1" onCompleted={onCompleted} />,
    );
    await waitFor(() => expect(startGuidedMock).toHaveBeenCalled());
    // Observe a live wizard FIRST so the observed-transition guard arms.
    useSessionStore.setState({
      guidedSession: guidedSessionPayload(null),
    } as never);
    await waitFor(() =>
      expect(useSessionStore.getState().guidedSession).not.toBeNull(),
    );
    useSessionStore.setState({
      guidedSession: guidedSessionPayload("completed"),
    } as never);
    await waitFor(() => expect(onCompleted).toHaveBeenCalledWith("sess-1"));
  });

  it("does NOT call onCompleted when mounted onto an already-completed session (back-nav GET)", async () => {
    // The back-nav path: startGuided resolves the PERSISTED completed guided
    // session WITHOUT the shell ever seeing a live (non-completed) state. This
    // is the exact bug — re-firing onCompleted here bounced the user back to
    // run. The observed-transition guard must suppress it.
    const onCompleted = vi.fn();
    startGuidedMock.mockImplementation(async () => {
      // Mirror sessionStore.startGuided: it sets guidedSession on the store.
      // Here the persisted session is already terminal=completed.
      useSessionStore.setState({
        guidedSession: guidedSessionPayload("completed"),
      } as never);
    });
    render(
      <TutorialGuidedShell sessionId="sess-1" onCompleted={onCompleted} />,
    );
    await waitFor(() => expect(startGuidedMock).toHaveBeenCalled());
    await waitFor(() =>
      expect(useSessionStore.getState().guidedSession).not.toBeNull(),
    );
    // Give the completion effect a chance to (wrongly) fire, then assert it did
    // not. The shell never observed a live wizard, so onCompleted stays unfired.
    await Promise.resolve();
    expect(onCompleted).not.toHaveBeenCalled();
  });

  it("fires onCompleted exactly once across a live->completed transition under StrictMode", async () => {
    // StrictMode double-invokes mount effects; completedRef/sawActiveRef must
    // make onCompleted fire exactly once for a single observed completion.
    const onCompleted = vi.fn();
    render(
      <StrictMode>
        <TutorialGuidedShell sessionId="sess-1" onCompleted={onCompleted} />
      </StrictMode>,
    );
    await waitFor(() => expect(startGuidedMock).toHaveBeenCalled());
    useSessionStore.setState({
      guidedSession: guidedSessionPayload(null),
    } as never);
    await waitFor(() =>
      expect(useSessionStore.getState().guidedSession).not.toBeNull(),
    );
    useSessionStore.setState({
      guidedSession: guidedSessionPayload("completed"),
    } as never);
    await waitFor(() => expect(onCompleted).toHaveBeenCalledWith("sess-1"));
    expect(onCompleted).toHaveBeenCalledTimes(1);
  });

  it("does NOT call onCompleted when the guided session exits to freeform (F-FE2)", async () => {
    // exited_to_freeform is a terminal kind, but leaving the wizard for freeform
    // is not a graduation: only terminal.kind === "completed" hands off.
    const onCompleted = vi.fn();
    render(
      <TutorialGuidedShell sessionId="sess-1" onCompleted={onCompleted} />,
    );
    await waitFor(() => expect(startGuidedMock).toHaveBeenCalled());
    useSessionStore.setState({
      guidedSession: guidedSessionPayload(null),
    } as never);
    await waitFor(() =>
      expect(useSessionStore.getState().guidedSession).not.toBeNull(),
    );
    useSessionStore.setState({
      guidedSession: guidedSessionPayload("exited_to_freeform"),
    } as never);
    await Promise.resolve();
    expect(onCompleted).not.toHaveBeenCalled();
  });

  it("calls onExited when an observed live wizard exits to freeform (elspeth-61591e64bb)", async () => {
    // The wire-stage "Exit to freeform" button is reachable in tutorial mode
    // and produces a real exited_to_freeform terminal. Without a hand-off the
    // shell stays mounted over a terminal session and ChatPanel dead-ends on
    // the "Preparing your guided pipeline…" placeholder.
    const onCompleted = vi.fn();
    const onExited = vi.fn();
    render(
      <TutorialGuidedShell
        sessionId="sess-1"
        onCompleted={onCompleted}
        onExited={onExited}
      />,
    );
    await waitFor(() => expect(startGuidedMock).toHaveBeenCalled());
    useSessionStore.setState({
      guidedSession: guidedSessionPayload(null),
    } as never);
    await waitFor(() =>
      expect(useSessionStore.getState().guidedSession).not.toBeNull(),
    );
    useSessionStore.setState({
      guidedSession: guidedSessionPayload("exited_to_freeform"),
    } as never);
    await waitFor(() => expect(onExited).toHaveBeenCalledWith("sess-1"));
    expect(onExited).toHaveBeenCalledTimes(1);
    expect(onCompleted).not.toHaveBeenCalled();
  });

  it("does NOT call onExited when mounted onto an already-exited session", async () => {
    // Mirror of the back-nav guard for completions: a persisted
    // exited_to_freeform session resolved straight from startGuided was never
    // a live wizard under this mount, so the shell must not re-fire the exit
    // hand-off (which would re-PATCH the opt-out on every remount).
    const onExited = vi.fn();
    startGuidedMock.mockImplementation(async () => {
      useSessionStore.setState({
        guidedSession: guidedSessionPayload("exited_to_freeform"),
      } as never);
    });
    render(
      <TutorialGuidedShell
        sessionId="sess-1"
        onCompleted={vi.fn()}
        onExited={onExited}
      />,
    );
    await waitFor(() => expect(startGuidedMock).toHaveBeenCalled());
    await waitFor(() =>
      expect(useSessionStore.getState().guidedSession).not.toBeNull(),
    );
    await Promise.resolve();
    expect(onExited).not.toHaveBeenCalled();
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
