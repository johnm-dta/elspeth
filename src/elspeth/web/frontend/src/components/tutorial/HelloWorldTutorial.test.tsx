import { StrictMode } from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { HelloWorldTutorial } from "./HelloWorldTutorial";
import { TutorialTurn4Run } from "./TutorialTurn4Run";

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
    title: "hello-world (pending)",
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
  });

  it("renders the welcome bookend first", () => {
    render(<HelloWorldTutorial />);
    expect(
      screen.getByRole("heading", { name: /welcome/i }),
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

  it("tags the created tutorial session before entering guided", async () => {
    const api = await import("@/api/client");
    const user = userEvent.setup();
    render(<HelloWorldTutorial />);
    await user.click(screen.getByRole("button", { name: "Let's go" }));
    await waitFor(() =>
      expect(api.renameSession).toHaveBeenCalledWith(
        "sess-new",
        "hello-world (pending)",
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
          prompt="strict prompt"
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
      prompt: "strict prompt",
    });
    expect(signal).toBeInstanceOf(AbortSignal);
  });
});
