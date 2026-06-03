import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { resetStore } from "@/test/store-helpers";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { useSessionStore } from "@/stores/sessionStore";
import * as api from "@/api/client";
import { TutorialTurn7Graduation } from "./TutorialTurn7Graduation";

vi.mock("@/api/client", () => ({
  createSession: vi.fn(),
  fetchUserComposerPreferences: vi.fn(),
  startGuided: vi.fn(),
  updateUserComposerPreferences: vi.fn(),
}));

describe("TutorialTurn7Graduation", () => {
  beforeEach(() => {
    resetStore(usePreferencesStore);
    resetStore(useSessionStore);
    vi.clearAllMocks();
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "freeform",
      tutorialCompletedAt: null,
      tutorialCompleted: false,
    });
    vi.mocked(api.createSession).mockResolvedValue({
      id: "session-empty",
      title: "New session",
      created_at: "2026-05-19T12:30:00Z",
      updated_at: "2026-05-19T12:30:00Z",
    });
    vi.mocked(api.updateUserComposerPreferences).mockResolvedValue({
      default_mode: "freeform",
      banner_dismissed_at: null,
      tutorial_completed_at: "2026-05-19T12:35:00Z",
      updated_at: "2026-05-19T12:35:00Z",
    });
  });

  it("focuses the heading, emits the graduation event, and renders the learning bullets", async () => {
    const eventListener = vi.fn();
    window.addEventListener("tutorial_graduation_shown", eventListener);

    render(<TutorialTurn7Graduation onBack={() => undefined} />);

    const heading = screen.getByRole("heading", {
      name: "You're ready to use the composer.",
    });
    await waitFor(() => expect(heading).toHaveFocus());
    expect(eventListener).toHaveBeenCalledTimes(1);
    expect(screen.getByText("What you built is AI-generated.")).toBeInTheDocument();
    expect(screen.getByText("Read before you run.")).toBeInTheDocument();
    expect(screen.getByText("Ask Elspeth.")).toBeInTheDocument();
    expect(screen.getByText("LLMs are confident even when they're wrong.")).toBeInTheDocument();

    window.removeEventListener("tutorial_graduation_shown", eventListener);
  });

  it("marks the tutorial graduated before creating the composer session", async () => {
    const user = userEvent.setup();
    render(<TutorialTurn7Graduation onBack={() => undefined} />);

    await user.click(
      screen.getByRole("button", { name: "Take me to the composer" }),
    );

    await waitFor(() => {
      expect(api.updateUserComposerPreferences).toHaveBeenCalledWith({
        tutorial_completed_at: expect.any(String),
      });
    });
    expect(api.createSession).toHaveBeenCalledTimes(1);
    expect(usePreferencesStore.getState().tutorialCompleted).toBe(true);
    expect(useSessionStore.getState().activeSessionId).toBe("session-empty");
  });

  it("shows a role alert on completion failure and keeps Back usable", async () => {
    const user = userEvent.setup();
    const onBack = vi.fn();
    vi.mocked(api.updateUserComposerPreferences).mockRejectedValueOnce(
      new Error("network down"),
    );
    render(<TutorialTurn7Graduation onBack={onBack} />);

    await user.click(
      screen.getByRole("button", { name: "Take me to the composer" }),
    );

    expect(await screen.findByRole("alert")).toHaveTextContent("network down");
    expect(api.createSession).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Back" }));
    expect(onBack).toHaveBeenCalledTimes(1);
  });

  it("shows a role alert when the fresh composer session cannot be created after graduation is saved", async () => {
    const user = userEvent.setup();
    vi.mocked(api.createSession).mockRejectedValueOnce(
      new Error("session service down"),
    );
    render(<TutorialTurn7Graduation onBack={() => undefined} />);

    await user.click(
      screen.getByRole("button", { name: "Take me to the composer" }),
    );

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Failed to create session. Please try again.",
    );
    expect(api.updateUserComposerPreferences).toHaveBeenCalledWith({
      tutorial_completed_at: expect.any(String),
    });
    expect(usePreferencesStore.getState().tutorialCompleted).toBe(false);
    expect(useSessionStore.getState().activeSessionId).toBeNull();
  });
});
