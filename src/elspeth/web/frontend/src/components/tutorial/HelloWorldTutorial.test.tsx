import { StrictMode } from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { HelloWorldTutorial } from "./HelloWorldTutorial";
import { TutorialTurn4Run } from "./TutorialTurn4Run";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { resetStore } from "@/test/store-helpers";

vi.mock("@/api/client", () => ({
  deleteTutorialOrphans: vi.fn().mockResolvedValue({ deleted_count: 0 }),
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
vi.mock("./TutorialGuidedShell", () => ({
  TutorialGuidedShell: ({
    onCompleted,
  }: {
    onCompleted: (s: string) => void;
  }) => (
    <button type="button" onClick={() => onCompleted("sess-new")}>
      finish-guided
    </button>
  ),
}));

describe("HelloWorldTutorial staged flow", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore(usePreferencesStore);
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

describe("HelloWorldTutorial — server-persisted resume (elspeth-918f4434b3)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore(usePreferencesStore);
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
