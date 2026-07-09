import { StrictMode } from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { HelloWorldTutorial } from "./HelloWorldTutorial";
import { TutorialTurn4Run } from "./TutorialTurn4Run";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { useSessionStore } from "@/stores/sessionStore";
import { resetStore } from "@/test/store-helpers";
import type { GuidedSession } from "@/types/guided";

vi.mock("@/api/client", () => ({
  deleteTutorialOrphans: vi.fn().mockResolvedValue({ deleted_count: 0 }),
  // Mount-time resume validation lists live sessions; default includes the
  // canonical resume session so the existing resume tests read as "alive".
  fetchSessions: vi.fn().mockResolvedValue([
    {
      id: "sess-resume",
      title: "First-run tutorial (in progress)",
      created_at: "2026-05-19T12:00:00Z",
      updated_at: "2026-05-19T12:00:00Z",
    },
  ]),
  createSession: vi.fn().mockResolvedValue({
    id: "sess-new",
    title: "New session",
    created_at: "2026-05-19T12:00:00Z",
    updated_at: "2026-05-19T12:00:00Z",
  }),
  renameSession: vi.fn().mockResolvedValue({
    id: "sess-new",
    title: "First-run tutorial (in progress)",
    created_at: "2026-05-19T12:00:00Z",
    updated_at: "2026-05-19T12:00:00Z",
  }),
  // TutorialTurn4Run reaches the real api.runTutorialPipeline in the relocated
  // StrictMode dedup test below, so it must be mocked even though the staged
  // flow tests stop at the mocked guided shell.
  runTutorialPipeline: vi.fn().mockResolvedValue({
    run_id: "run-1",
    output: {
      source_data_hash: "a7f3e2fullhash",
      rows: [{ url: "dta.gov.au", score: 9, rationale: "bold" }],
      discarded_row_count: 0,
    },
  }),
  sendTutorialAbandonBeacon: vi.fn(),
  // The chrome "Exit tutorial" control abandons an in-flight run via
  // TutorialTurn4Run's abandonTutorialRun, which fires this.
  cancelTutorialRun: vi.fn().mockResolvedValue({ cancelled: true }),
  // The tutorial-persistence slice (elspeth-918f4434b3): the component
  // persists stage transitions through preferencesStore, which calls this.
  // Body-aware echo mirroring the backend upsert (supplied fields land in
  // the response; completion clears progress server-side).
  updateUserComposerPreferences: vi.fn(async (body: Record<string, unknown>) => ({
    default_mode: "guided",
    banner_dismissed_at: null,
    tutorial_completed_at: body.tutorial_completed_at ?? null,
    tutorial_stage: body.tutorial_completed_at ? null : (body.tutorial_stage ?? null),
    tutorial_session_id: body.tutorial_completed_at
      ? null
      : (body.tutorial_session_id ?? null),
    tutorial_run_id: body.tutorial_completed_at ? null : (body.tutorial_run_id ?? null),
    tutorial_source_data_hash: body.tutorial_completed_at
      ? null
      : (body.tutorial_source_data_hash ?? null),
    updated_at: "2026-07-02T00:00:00Z",
  })),
  fetchUserComposerPreferences: vi.fn(),
  // The resumed-audit test mounts the real TutorialTurn5AuditStory.
  getRunAuditSummary: vi.fn().mockResolvedValue({
    run_id: "run-resume",
    session_id: "sess-resume",
    llm_call_count: 3,
    source_data_hash: "hash-resume",
    started_at: "2026-07-02T00:00:00Z",
    plugin_versions: {},
    seeded_from_cache: false,
    cache_key: null,
  }),
}));

// Replace the embedded guided surface with a one-button stub that fires the
// completion callback; the real ChatPanel guided behaviour is covered by
// TutorialGuidedShell.test.tsx.
// When set, the stub fires onSessionMissing TWICE on mount with its session
// id — simulating the live race where the shell's guided/start 404 handler
// and the mount-time membership check both detect the same dead resume.
let stubShellReportsSessionMissing = false;
// Per-test session id the stub hands to onCompleted/onExited. The run turn's
// StrictMode dedupe cache is module-level and keyed by sessionId, so tests
// that exercise the run step with a bespoke runTutorialPipeline mock must use
// a distinct id or they replay a previous test's cached run.
let stubGuidedSessionId = "sess-new";
vi.mock("./TutorialGuidedShell", async () => {
  const { useEffect } = await import("react");
  return {
    TutorialGuidedShell: ({
      sessionId,
      onCompleted,
      onExited,
      onSessionMissing,
    }: {
      sessionId: string;
      onCompleted: (s: string) => void;
      onExited?: (s: string) => void;
      onSessionMissing?: (deadSessionId: string) => void;
    }) => {
      useEffect(() => {
        if (stubShellReportsSessionMissing) {
          onSessionMissing?.(sessionId);
          onSessionMissing?.(sessionId);
        }
        // Mount-only, mirroring the real shell's start effect.
        // eslint-disable-next-line react-hooks/exhaustive-deps
      }, []);
      return (
        <>
          <button type="button" onClick={() => onCompleted(stubGuidedSessionId)}>
            finish-guided
          </button>
          <button type="button" onClick={() => onExited?.(stubGuidedSessionId)}>
            exit-guided
          </button>
        </>
      );
    },
  };
});

describe("HelloWorldTutorial staged flow", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore(usePreferencesStore);
    stubShellReportsSessionMissing = false;
    stubGuidedSessionId = "sess-new";
  });

  it("renders the welcome bookend first", () => {
    render(<HelloWorldTutorial />);
    expect(
      screen.getByRole("heading", { name: /welcome/i }),
    ).toBeInTheDocument();
  });

  it("labels the tutorial progress row visibly so it reads as a different hierarchy from the build stepper", async () => {
    // elspeth-d75756fa2c: the 5 unlabeled dots sat directly above the guided
    // 5-chip build stepper and read as a broken duplicate of it. The visible
    // "Tutorial · <stage>" label is aria-hidden — the existing sr-only "Step N
    // of M" line stays the AT signal (the ARIA was already right).
    const user = userEvent.setup();
    render(<HelloWorldTutorial />);
    const label = screen.getByText("Tutorial · Welcome — step 1 of 5");
    expect(label).toHaveAttribute("aria-hidden", "true");

    await user.click(screen.getByRole("button", { name: "Let's go" }));
    expect(
      await screen.findByText("Tutorial · Build — step 2 of 5"),
    ).toBeInTheDocument();
  });

  it("runs orphan cleanup on mount", async () => {
    const api = await import("@/api/client");
    render(<HelloWorldTutorial />);
    expect(api.deleteTutorialOrphans).toHaveBeenCalledTimes(1);
  });

  it("advances welcome -> guided -> run on guided completion", async () => {
    const user = userEvent.setup();
    render(<HelloWorldTutorial />);
    await user.click(screen.getByRole("button", { name: "Let's go" }));
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "finish-guided" }),
      ).toBeInTheDocument(),
    );
    await user.click(screen.getByRole("button", { name: "finish-guided" }));
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "finish-guided" }),
      ).toBeNull(),
    );
  });

  it("renders no Back affordance on the run turn (consumed guided wizard is non-returnable)", async () => {
    // previousStep(run) is null, so HelloWorldTutorial passes no onBack to the
    // run turn and TutorialTurn4Run renders no Back button. This is the wiring
    // half of the fix: Back from run must not remount the completed guided
    // wizard (which would re-fire onCompleted and bounce the user back to run).
    const user = userEvent.setup();
    render(<HelloWorldTutorial />);
    await user.click(screen.getByRole("button", { name: "Let's go" }));
    await user.click(
      await screen.findByRole("button", { name: "finish-guided" }),
    );
    // The run turn fetches via the mocked runTutorialPipeline and renders its
    // result row ("bold" rationale) plus the primary "continue" button.
    expect(await screen.findByText("bold")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Back/ })).toBeNull();
  });

  it("tags the created tutorial session before entering guided", async () => {
    const api = await import("@/api/client");
    const user = userEvent.setup();
    render(<HelloWorldTutorial />);
    await user.click(screen.getByRole("button", { name: "Let's go" }));
    await waitFor(() =>
      expect(api.renameSession).toHaveBeenCalledWith(
        "sess-new",
        "First-run tutorial (in progress)",
      ),
    );
    const createOrder = vi.mocked(api.createSession).mock
      .invocationCallOrder[0];
    const renameOrder = vi.mocked(api.renameSession).mock
      .invocationCallOrder[0];
    expect(createOrder).toBeLessThan(renameOrder);
  });

  it("surfaces createSession failure instead of stalling on welcome", async () => {
    const api = await import("@/api/client");
    vi.mocked(api.createSession).mockRejectedValueOnce(
      new Error("session service down"),
    );
    const user = userEvent.setup();
    render(<HelloWorldTutorial />);
    await user.click(screen.getByRole("button", { name: "Let's go" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "session service down",
    );
    // Still on welcome — the guided shell never mounted.
    expect(
      screen.getByRole("heading", { name: /welcome/i }),
    ).toBeInTheDocument();
  });

  it("preflights composer availability before creating a guided tutorial session", async () => {
    const api = await import("@/api/client");
    render(
      <HelloWorldTutorial
        composerAvailable={false}
        composerUnavailableReason="Composer model openrouter/openai/gpt-4o is unavailable: missing OPENROUTER_API_KEY."
      />,
    );

    const start = screen.getByRole("button", { name: "Let's go" });
    expect(start).toBeDisabled();
    expect(screen.getByRole("alert")).toHaveTextContent(
      "missing OPENROUTER_API_KEY",
    );
    expect(
      screen.getByRole("button", { name: "Skip the tutorial" }),
    ).toBeEnabled();
    expect(api.createSession).not.toHaveBeenCalled();
  });

  // Relocated from the old big-bang test: TutorialTurn4Run dedups the run
  // request under StrictMode's double-invoke. This is the only coverage of
  // that behaviour, so it rides along with HelloWorldTutorial's suite rather
  // than being dropped with the removed turns.
  it("settles the run turn under React StrictMode without duplicating the run request", async () => {
    const api = await import("@/api/client");
    render(
      <StrictMode>
        <TutorialTurn4Run
          sessionId="strict-session"
          onCompleted={() => undefined}
          onCancelled={() => undefined}
          onBack={() => undefined}
        />
      </StrictMode>,
    );

    expect(await screen.findByText("bold")).toBeInTheDocument();
    expect(api.runTutorialPipeline).toHaveBeenCalledTimes(1);
    const [body, signal] = vi.mocked(api.runTutorialPipeline).mock.calls[0];
    expect(body).toEqual({
      session_id: "strict-session",
    });
    expect(signal).toBeInstanceOf(AbortSignal);
  });
});

describe("HelloWorldTutorial — abandon beacon (F4)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore(usePreferencesStore);
    stubGuidedSessionId = "sess-new";
  });

  it("does not fire the abandon beacon on pagehide while still on the welcome bookend", async () => {
    const api = await import("@/api/client");
    render(<HelloWorldTutorial />);
    // Nothing has started yet — there is no in-progress tutorial to abandon.
    window.dispatchEvent(new Event("pagehide"));
    expect(api.sendTutorialAbandonBeacon).not.toHaveBeenCalled();
  });

  it("fires the abandon beacon on pagehide once the tutorial is in progress", async () => {
    const api = await import("@/api/client");
    const user = userEvent.setup();
    render(<HelloWorldTutorial />);
    await user.click(screen.getByRole("button", { name: "Let's go" }));
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "finish-guided" }),
      ).toBeInTheDocument(),
    );

    window.dispatchEvent(new Event("pagehide"));

    expect(api.sendTutorialAbandonBeacon).toHaveBeenCalledTimes(1);
  });

  it("does not fire the abandon beacon on pagehide once graduation is reached (skip path)", async () => {
    const api = await import("@/api/client");
    const user = userEvent.setup();
    render(<HelloWorldTutorial />);
    await user.click(screen.getByRole("button", { name: "Skip the tutorial" }));
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: "You're ready to use the composer." }),
      ).toBeInTheDocument(),
    );

    window.dispatchEvent(new Event("pagehide"));

    // Skip lands on the same terminal `graduation` step every completing path
    // reaches — the learner saw the tutorial through, so this is not an
    // abandon, regardless of which door they left through.
    expect(api.sendTutorialAbandonBeacon).not.toHaveBeenCalled();
  });

  it("removes its pagehide listener on unmount so a later pagehide cannot fire the beacon", async () => {
    const api = await import("@/api/client");
    const user = userEvent.setup();
    const { unmount } = render(<HelloWorldTutorial />);
    await user.click(screen.getByRole("button", { name: "Let's go" }));
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "finish-guided" }),
      ).toBeInTheDocument(),
    );

    unmount();
    window.dispatchEvent(new Event("pagehide"));

    expect(api.sendTutorialAbandonBeacon).not.toHaveBeenCalled();
  });
});

describe("HelloWorldTutorial — exit to freeform (elspeth-61591e64bb)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore(usePreferencesStore);
    stubGuidedSessionId = "sess-new";
  });

  it("persists the exit opt-out with the exit discriminator when guided exits to freeform", async () => {
    const api = await import("@/api/client");
    const user = userEvent.setup();
    render(<HelloWorldTutorial />);
    await user.click(screen.getByRole("button", { name: "Let's go" }));
    await user.click(await screen.findByRole("button", { name: "exit-guided" }));
    await waitFor(() =>
      expect(api.updateUserComposerPreferences).toHaveBeenCalledWith({
        tutorial_completed_at: expect.any(String),
        tutorial_completed_via: "exit",
      }),
    );
    // Unlike skip (which keeps the graduation farewell mounted), exit
    // publishes locally so App's showTutorial gate unmounts the shell and
    // the learner lands in freeform on the same session.
    await waitFor(() =>
      expect(usePreferencesStore.getState().tutorialCompleted).toBe(true),
    );
  });

  it("renders an Exit tutorial control on the guided step that persists the same opt-out", async () => {
    const api = await import("@/api/client");
    const user = userEvent.setup();
    render(<HelloWorldTutorial />);
    await user.click(screen.getByRole("button", { name: "Let's go" }));
    await user.click(
      await screen.findByRole("button", { name: "Exit tutorial" }),
    );
    await waitFor(() =>
      expect(api.updateUserComposerPreferences).toHaveBeenCalledWith(
        expect.objectContaining({
          tutorial_completed_at: expect.any(String),
          tutorial_completed_via: "exit",
        }),
      ),
    );
    await waitFor(() =>
      expect(usePreferencesStore.getState().tutorialCompleted).toBe(true),
    );
  });

  it("renders the Exit tutorial control on the run step", async () => {
    const user = userEvent.setup();
    render(<HelloWorldTutorial />);
    await user.click(screen.getByRole("button", { name: "Let's go" }));
    await user.click(
      await screen.findByRole("button", { name: "finish-guided" }),
    );
    expect(await screen.findByText("bold")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Exit tutorial" }),
    ).toBeInTheDocument();
  });

  it("renders the Exit tutorial control on the resumed audit step", async () => {
    usePreferencesStore.setState({
      loaded: true,
      tutorialStage: "run",
      tutorialSessionId: "sess-resume",
      tutorialRunId: "run-resume",
      tutorialSourceDataHash: "hash-resume",
    });
    render(<HelloWorldTutorial />);
    expect(
      await screen.findByRole("heading", { name: "This is the audit story." }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Exit tutorial" }),
    ).toBeInTheDocument();
  });

  it("offers no Exit tutorial control on the welcome bookend (Skip is the welcome exit)", () => {
    render(<HelloWorldTutorial />);
    expect(
      screen.getByRole("button", { name: "Skip the tutorial" }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Exit tutorial" })).toBeNull();
  });

  it("offers no Exit tutorial control on graduation (the finish CTA is the exit)", () => {
    usePreferencesStore.setState({
      loaded: true,
      tutorialStage: "graduation",
      tutorialSessionId: "sess-resume",
      tutorialRunId: "run-resume",
      tutorialSourceDataHash: "hash-resume",
    });
    render(<HelloWorldTutorial />);
    expect(
      screen.getByRole("heading", { name: "You're ready to use the composer." }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Exit tutorial" })).toBeNull();
  });

  it("Exit tutorial during Build terminates the live guided session", async () => {
    // Without this hand-off the non-terminal guidedSession survives the
    // shell unmount inside sessionStore, and the freeform ChatPanel's
    // discriminator re-renders the guided workspace — the learner has to
    // exit guided a second time instead of landing in freeform as promised.
    const exitToFreeform = vi.fn().mockResolvedValue(undefined);
    const originalExit = useSessionStore.getState().exitToFreeform;
    useSessionStore.setState({
      guidedSession: { terminal: null } as unknown as GuidedSession,
      exitToFreeform,
    });
    try {
      const user = userEvent.setup();
      render(<HelloWorldTutorial />);
      await user.click(screen.getByRole("button", { name: "Let's go" }));
      await user.click(
        await screen.findByRole("button", { name: "Exit tutorial" }),
      );
      expect(exitToFreeform).toHaveBeenCalledTimes(1);
    } finally {
      useSessionStore.setState({
        exitToFreeform: originalExit,
        guidedSession: null,
      });
    }
  });

  it("adds the renamed tutorial session to the header session list before guided starts", async () => {
    const user = userEvent.setup();
    render(<HelloWorldTutorial />);

    await user.click(screen.getByRole("button", { name: "Let's go" }));

    await waitFor(() => {
      expect(useSessionStore.getState().sessions).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            id: "sess-new",
            title: "First-run tutorial (in progress)",
          }),
        ]),
      );
    });
  });

  it("Exit tutorial leaves an already-exited-to-freeform guided session alone", async () => {
    // The wizard-path hand-off (onExited) reaches the same handler with the
    // terminal already set to exited_to_freeform — firing the control signal
    // again would be a duplicate respond POST that the backend 409s (only
    // kind=COMPLETED is exempt from the terminal-rejection; guided.py:1190).
    const exitToFreeform = vi.fn().mockResolvedValue(undefined);
    const originalExit = useSessionStore.getState().exitToFreeform;
    useSessionStore.setState({
      guidedSession: {
        terminal: { kind: "exited_to_freeform", reason: "user_pressed_exit" },
      } as unknown as GuidedSession,
      exitToFreeform,
    });
    try {
      const user = userEvent.setup();
      render(<HelloWorldTutorial />);
      await user.click(screen.getByRole("button", { name: "Let's go" }));
      await user.click(
        await screen.findByRole("button", { name: "Exit tutorial" }),
      );
      expect(exitToFreeform).not.toHaveBeenCalled();
    } finally {
      useSessionStore.setState({
        exitToFreeform: originalExit,
        guidedSession: null,
      });
    }
  });

  it("Exit tutorial after the guided wizard COMPLETED hands off to freeform", async () => {
    // elspeth-e2c3dba6b5 review P2: a COMPLETED guided session survives the
    // shell unmount, and ChatPanel's discriminator checks `completed` FIRST —
    // so without this hand-off the learner lands on CompletionSummary and must
    // click "Open freeform editor" (which itself just calls exitToFreeform) to
    // reach freeform, contradicting the Exit control's "freeform NOW" promise.
    // The backend accepts exit_to_freeform from kind=COMPLETED (guided.py:1222),
    // transitioning it to exited_to_freeform so the surface falls through.
    const exitToFreeform = vi.fn().mockResolvedValue(undefined);
    const originalExit = useSessionStore.getState().exitToFreeform;
    useSessionStore.setState({
      guidedSession: {
        terminal: { kind: "completed", reason: null },
      } as unknown as GuidedSession,
      exitToFreeform,
    });
    try {
      const user = userEvent.setup();
      render(<HelloWorldTutorial />);
      await user.click(screen.getByRole("button", { name: "Let's go" }));
      await user.click(
        await screen.findByRole("button", { name: "Exit tutorial" }),
      );
      expect(exitToFreeform).toHaveBeenCalledTimes(1);
    } finally {
      useSessionStore.setState({
        exitToFreeform: originalExit,
        guidedSession: null,
      });
    }
  });

  it("Exit tutorial during an in-flight run aborts the fetch and fires the server-side cancel", async () => {
    const api = await import("@/api/client");
    // Distinct session id: the run cache is module-level (see the stub note).
    stubGuidedSessionId = "sess-exit-mid-run";
    // A run that never settles — the exit must not depend on it finishing.
    vi.mocked(api.runTutorialPipeline).mockImplementationOnce(
      () => new Promise(() => {}),
    );
    const user = userEvent.setup();
    render(<HelloWorldTutorial />);
    await user.click(screen.getByRole("button", { name: "Let's go" }));
    await user.click(
      await screen.findByRole("button", { name: "finish-guided" }),
    );
    await user.click(screen.getByRole("button", { name: "Exit tutorial" }));

    const [, signal] = vi.mocked(api.runTutorialPipeline).mock.calls[0];
    expect((signal as AbortSignal).aborted).toBe(true);
    expect(api.cancelTutorialRun).toHaveBeenCalledWith("sess-exit-mid-run");
  });

  it("Exit tutorial after the run completes does not cancel the finished run", async () => {
    const api = await import("@/api/client");
    stubGuidedSessionId = "sess-exit-after-run";
    const user = userEvent.setup();
    render(<HelloWorldTutorial />);
    await user.click(screen.getByRole("button", { name: "Let's go" }));
    await user.click(
      await screen.findByRole("button", { name: "finish-guided" }),
    );
    expect(await screen.findByText("bold")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Exit tutorial" }));

    expect(api.cancelTutorialRun).not.toHaveBeenCalled();
  });
});

describe("HelloWorldTutorial — server-persisted resume (elspeth-918f4434b3)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore(usePreferencesStore);
    stubGuidedSessionId = "sess-new";
  });

  it("persists the guided stage + session on Start", async () => {
    const api = await import("@/api/client");
    const user = userEvent.setup();
    render(<HelloWorldTutorial />);
    await user.click(screen.getByRole("button", { name: "Let's go" }));
    await waitFor(() =>
      expect(api.updateUserComposerPreferences).toHaveBeenCalledWith({
        tutorial_stage: "guided",
        tutorial_session_id: "sess-new",
        tutorial_run_id: null,
        tutorial_source_data_hash: null,
      }),
    );
  });

  it("persists the run stage when guided completes", async () => {
    const api = await import("@/api/client");
    const user = userEvent.setup();
    render(<HelloWorldTutorial />);
    await user.click(screen.getByRole("button", { name: "Let's go" }));
    await user.click(
      await screen.findByRole("button", { name: "finish-guided" }),
    );
    await waitFor(() =>
      expect(api.updateUserComposerPreferences).toHaveBeenCalledWith(
        expect.objectContaining({
          tutorial_stage: "run",
          tutorial_session_id: "sess-new",
        }),
      ),
    );
  });

  it("persists the skip opt-out IMMEDIATELY on the first click", async () => {
    const api = await import("@/api/client");
    const user = userEvent.setup();
    render(<HelloWorldTutorial />);
    await user.click(screen.getByRole("button", { name: "Skip the tutorial" }));
    // The opt-out lands on the FIRST click — not on the follow-up
    // "Take me to the composer" click (which is never made here).
    await waitFor(() =>
      expect(api.updateUserComposerPreferences).toHaveBeenCalledWith({
        tutorial_completed_at: expect.any(String),
      }),
    );
    // No stage write for the skip path: the completion persist above
    // clears the resume state server-side.
    const bodies = vi.mocked(api.updateUserComposerPreferences).mock.calls.map(
      ([body]) => body,
    );
    expect(bodies.some((body) => "tutorial_stage" in body)).toBe(false);
    // The graduation card stays mounted (publishLocally=false): the skip
    // must not yank the farewell out from under the user.
    expect(
      screen.getByRole("heading", { name: "You're ready to use the composer." }),
    ).toBeInTheDocument();
  });

  it("resumes at guided with the persisted session — no orphan sweep, no restart", async () => {
    const api = await import("@/api/client");
    usePreferencesStore.setState({
      loaded: true,
      tutorialStage: "guided",
      tutorialSessionId: "sess-resume",
    });
    render(<HelloWorldTutorial />);
    // The guided shell (stubbed) mounts directly — not the Welcome bookend.
    expect(
      screen.getByRole("button", { name: "finish-guided" }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: /welcome/i })).toBeNull();
    // The resumable session must NOT be renamed "abandoned-..." on reload:
    // orphan cleanup is a fresh-entry-only sweep.
    expect(api.deleteTutorialOrphans).not.toHaveBeenCalled();
    // And no session re-creation — the persisted session is reused.
    expect(api.createSession).not.toHaveBeenCalled();
  });

  it("falls back to a fresh Welcome when the resumed session no longer exists", async () => {
    // The persisted resume fields can outlive their session (orphan sweep,
    // archive, prerelease wipe). Without recovery the guided stage dead-ends
    // on "Session not found" with NO affordance (skip/exit are suppressed
    // past Welcome) — the operator-observed blank-page failure.
    const api = await import("@/api/client");
    // Once, not a standing override: clearAllMocks clears CALLS but keeps
    // implementations, so a standing mockResolvedValue([]) here would poison
    // the sibling resume tests into the recovery path.
    vi.mocked(api.fetchSessions).mockResolvedValueOnce([]);
    usePreferencesStore.setState({
      loaded: true,
      tutorialStage: "guided",
      tutorialSessionId: "sess-resume",
    });
    render(<HelloWorldTutorial />);
    // Recovery lands on the Welcome bookend, not the guided shell.
    expect(
      await screen.findByRole("heading", { name: /welcome/i }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "finish-guided" })).toBeNull();
    // And the stale server-side resume fields are cleared (welcome → all-null).
    await waitFor(() => {
      const bodies = vi
        .mocked(api.updateUserComposerPreferences)
        .mock.calls.map(([body]) => body);
      expect(
        bodies.some(
          (body) =>
            "tutorial_stage" in body && body.tutorial_stage === null,
        ),
      ).toBe(true);
    });
  });

  it("recovery is idempotent per dead id and releases the app-level session binding", async () => {
    // Live-observed residue of the dead-resume recovery: the shell's
    // guided/start 404 handler and the mount-time membership check RACE and
    // both detect the same dead session — the warning double-logged — and
    // activeSessionId stayed bound to the corpse, so InlineRunResults kept
    // polling /runs (404) while the user sat at Welcome.
    const { useSessionStore } = await import("@/stores/sessionStore");
    const warnSpy = vi
      .spyOn(console, "warn")
      .mockImplementation(() => undefined);
    stubShellReportsSessionMissing = true;
    useSessionStore.setState({ activeSessionId: "sess-resume" });
    usePreferencesStore.setState({
      loaded: true,
      tutorialStage: "guided",
      tutorialSessionId: "sess-resume",
    });
    render(<HelloWorldTutorial />);
    expect(
      await screen.findByRole("heading", { name: /welcome/i }),
    ).toBeInTheDocument();
    // The dead binding is released — pollers keyed on activeSessionId stop.
    expect(useSessionStore.getState().activeSessionId).toBeNull();
    // Double detection of the same dead id recovers ONCE.
    const recoveryWarns = warnSpy.mock.calls.filter(([msg]) =>
      String(msg).includes("persisted resume session no longer exists"),
    );
    expect(recoveryWarns).toHaveLength(1);
    warnSpy.mockRestore();
  });

  it("resumes a completed run at the audit step without re-executing", async () => {
    const api = await import("@/api/client");
    usePreferencesStore.setState({
      loaded: true,
      tutorialStage: "run",
      tutorialSessionId: "sess-resume",
      tutorialRunId: "run-resume",
      tutorialSourceDataHash: "hash-resume",
    });
    render(<HelloWorldTutorial />);
    expect(
      await screen.findByRole("heading", { name: "This is the audit story." }),
    ).toBeInTheDocument();
    // Zero re-execution: the run identity came from the persisted fields.
    expect(api.runTutorialPipeline).not.toHaveBeenCalled();
    // The resumed audit suppresses Back — there is no in-memory run cache,
    // so Back into the run turn would silently re-fire the pipeline.
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Continue" })).toBeInTheDocument(),
    );
    expect(screen.queryByRole("button", { name: /back/i })).toBeNull();
  });

  it("resumes at graduation once the graduation card has been shown", async () => {
    usePreferencesStore.setState({
      loaded: true,
      tutorialStage: "graduation",
      tutorialSessionId: "sess-resume",
      tutorialRunId: "run-resume",
      tutorialSourceDataHash: "hash-resume",
    });
    render(<HelloWorldTutorial />);
    expect(
      screen.getByRole("heading", { name: "You're ready to use the composer." }),
    ).toBeInTheDocument();
  });

  it("does not fire a progress write when the persisted state already matches", async () => {
    const api = await import("@/api/client");
    usePreferencesStore.setState({
      loaded: true,
      tutorialStage: "graduation",
      tutorialSessionId: "sess-resume",
      tutorialRunId: "run-resume",
      tutorialSourceDataHash: "hash-resume",
    });
    render(<HelloWorldTutorial />);
    // The mount-time persist effect dedupes against the store's mirror of
    // the server row — a resume must not immediately re-PATCH it.
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: "You're ready to use the composer." }),
      ).toBeInTheDocument(),
    );
    expect(api.updateUserComposerPreferences).not.toHaveBeenCalled();
  });
});
